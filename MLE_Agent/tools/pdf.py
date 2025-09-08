from typing import Annotated
import modal
from modal import App, Image, Volume
from pydantic import Field
import os
import re
import time
import subprocess
from pathlib import Path
from .bash import BashContainer, _ModalBashSession
from .edit import EditContainer
from .shared import agent_volume

# Separate Modal app to avoid conflicts with any Sandbox app
app = App("mle-agent-pdf")


def run_marker_install():
    """Warm up marker models by executing a small job during image build."""
    arxiv_link = "https://arxiv.org/pdf/2002.00719"
    marker_path = "/root/data/marker"
    os.makedirs(marker_path, exist_ok=True)
    subprocess.run(["curl", arxiv_link, "-o", f"{marker_path}/input.pdf"], check=True)
    subprocess.run(
        [
            "marker",
            marker_path,
            "--output_format",
            "markdown",
            "--page_range",
            "1",
            "--output_dir",
            f"{marker_path}/output",
        ],
        check=True,
    )


# Build image with dependencies; copy repo into /root for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("ninja", "packaging", "wheel")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.8.0",
        "uv pip install --system 'transformers>=4.51.2,<4.54.0'",
        "uv pip install --system marker-pdf==1.9.0",
        "uv pip install --system datasets>=3.4.1",
        "uv pip install --system einops>=0.8.1",
        "uv pip install --system psutil>=6.1.1",
        "uv pip install --system hf-transfer",
    )
    .env(
        {
            "VIRTUAL_ENV": "/usr/local",
            "PATH": "/usr/local/bin:$PATH",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "GIT_USER_NAME": os.environ.get("GIT_USER_NAME", "Default Name"),
            "GIT_USER_EMAIL": os.environ.get("GIT_USER_EMAIL", "default@example.com"),
        }
    )
    .run_commands(
        "git config --global user.name \"$GIT_USER_NAME\"",
        "git config --global user.email \"$GIT_USER_EMAIL\"",
    )
    .uv_pip_install(
        'fastmcp',
        'mcp',
        'modal'
    )
    .run_function(run_marker_install)
    .run_commands("uv pip install --system modal")
    .add_local_dir(str(_REPO_ROOT), "/root", copy=True)
)


def split_markdown_into_pages(base_filename: str) -> list[str]:
    assert os.path.exists(base_filename), f"File {base_filename} does not exist"
    assert base_filename.endswith(".md"), "File is not a markdown file"
    with open(base_filename, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    page_pattern = r"\{(\d+)\}---PAGE---\n(.*?)(?=\{\d+\}---PAGE---|$)"
    page_matches = re.findall(page_pattern, markdown_content, re.DOTALL)

    created_files: list[str] = []
    for page_num, page_content in page_matches:
        page_content = page_content.strip()
        dir_ = os.path.dirname(base_filename)
        name = os.path.basename(base_filename)
        page_filename = f"{name.split('.')[0]}_page_{page_num}.md"
        page_filepath = os.path.join(dir_, page_filename)
        with open(page_filepath, "w", encoding="utf-8") as f:
            f.write(page_content)
        created_files.append(page_filepath)
    return created_files


def _sanitize_title(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"[^a-z0-9\-\.]+", "", t)
    return t[:120] or "paper"


def _parse_arxiv_info(url: str) -> tuple[str, str | None, str | None]:
    import re as _re

    def _fetch(url_: str) -> str:
        res = subprocess.run(["curl", "-sL", url_], check=True, capture_output=True)
        return res.stdout.decode("utf-8", errors="ignore")

    m_abs = _re.search(r"arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5}(v\d+)?)", url)
    if not m_abs:
        m_abs = _re.search(r"arxiv\.org/(abs|pdf)/([a-z\-]+/\d{7}(v\d+)?)", url)
    if m_abs:
        arxiv_id = m_abs.group(2)
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url_ = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        html = _fetch(abs_url)
        title = None
        mt = _re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', html, _re.I)
        if mt:
            title = mt.group(1).strip()
        if not title:
            tt = _re.search(r"<title>(.*?)</title>", html, _re.I | _re.S)
            if tt:
                title = tt.group(1).split("- arXiv")[0].strip()
        return pdf_url_, title, arxiv_id
    return url, None, None


def _guess_title_from_url(url: str) -> str | None:
    try:
        head = subprocess.run(["curl", "-sI", url], check=True, capture_output=True)
        headers = head.stdout.decode("utf-8", errors="ignore")
        m = re.search(r'filename=\"?([^\";]+)\"?', headers, re.I)
        if m:
            name = os.path.splitext(os.path.basename(m.group(1).strip()))[0]
            return name
    except Exception:
        pass
    try:
        name = os.path.splitext(os.path.basename(url.split("?", 1)[0]))[0]
        return re.sub(r"[_\-]+", " ", name).strip() or None
    except Exception:
        return None


def save_pdf_to_markdown(base_dir: str, dict_bytes: dict[str, bytes], title: str | None = None):
    title = _sanitize_title(title or "paper")
    root = Path(base_dir)
    dest_dir = root / "papers" / title
    os.makedirs(dest_dir, exist_ok=True)
    markdown_file = None
    for name, value in dict_bytes.items():
        out_path = dest_dir / name
        os.makedirs(out_path.parent, exist_ok=True)
        # Write all artifacts as raw bytes to avoid decode errors (images, etc.).
        with open(out_path, "wb") as f:
            f.write(value)
        if name.endswith(".md"):
            markdown_file = out_path
    if markdown_file:
        try:
            split_markdown_into_pages(str(markdown_file))
        except Exception:
            pass
        return str(markdown_file)
    return str(dest_dir)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 10,
    serialized=True,
    volumes={
        "/root/sandbox": agent_volume,
    },
)
def pdf_to_markdown(
    pdf_url: str,
    base_dir: str,
    page_range: str | None = None,
    save_remote: bool = True,
    force_redo: bool = False,
) -> str | tuple[dict[str, bytes], str]:
    os.makedirs(base_dir, exist_ok=True)
    import tempfile
    import torch
    import shutil

    # Prepare cache root
    CACHE_ROOT = Path("/root/sandbox/paper_cache")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_p = Path(temp_dir)
        output_dir = temp_dir_p / "output"
        pdf_path = temp_dir_p / "input.pdf"

        arxiv_pdf_url, arxiv_title, _ = _parse_arxiv_info(pdf_url)
        source_url = pdf_url
        pdf_url = arxiv_pdf_url

        # Try an early title guess to hit the cache before heavy work
        pre_title_guess = _sanitize_title(
            (arxiv_title or _guess_title_from_url(source_url) or "")
        ) if (arxiv_title or _guess_title_from_url(source_url)) else None

        if pre_title_guess:
            cached_dir = CACHE_ROOT / pre_title_guess
            if cached_dir.exists() and any(cached_dir.glob("*.md")) and not force_redo:
                if save_remote:
                    # Copy from cache into model directory and return a path
                    dest_dir = Path(base_dir) / "papers" / pre_title_guess
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(cached_dir, dest_dir, dirs_exist_ok=True)
                    md_file = next((p for p in dest_dir.glob("*.md")), None)
                    return str(md_file) if md_file else str(dest_dir)
                else:
                    # Return the cached files to be saved locally by caller
                    result_cached: dict[str, bytes] = {}
                    for fp in cached_dir.rglob("*"):
                        if fp.is_file():
                            result_cached[fp.name] = fp.read_bytes()
                    return result_cached, pre_title_guess

        subprocess.run(["curl", "-L", pdf_url, "-o", pdf_path], check=True)
        print(f"PDF path: {pdf_path}")
        print(f"Temp dir: {temp_dir_p} {os.listdir(temp_dir_p)}")

        assert torch.cuda.is_available(), "CUDA is not available"
        t0 = time.time()
        cmds = [
            "marker_single",
            pdf_path,
            "--output_format",
            "markdown",
            "--paginate_output",
            "--MarkdownRenderer_page_separator",
            "---PAGE---",
            "--output_dir",
            output_dir,
        ]
        if page_range:
            cmds.extend(["--page_range", page_range])
        subprocess.run(cmds, check=True)
        print(f"Marker took {time.time() - t0} seconds")
        print(f"Markdown files saved to {output_dir} {os.listdir(output_dir)}")

        result: dict[str, bytes] = {}
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path != pdf_path:
                with open(file_path, "rb") as f:
                    result[file_path.name] = f.read()

        md_guess = next((k for k in result.keys() if k.endswith(".md")), None)
        title_guess = (
            arxiv_title or _guess_title_from_url(source_url) or (md_guess or "paper").replace(".md", "")
        )
        title_guess = _sanitize_title(title_guess or "paper")

        # Save to cache regardless of save_remote so we don't redo work later
        cache_dir = CACHE_ROOT / title_guess
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, value in result.items():
            out_path = cache_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(value)

        if save_remote:
            output = save_pdf_to_markdown(base_dir, result, title_guess)
        else:
            output = result, title_guess
            
        print(f"os.listdir({base_dir}): {os.listdir(base_dir)}")
        return output


class PdfContainer:
    def __init__(self, bash_container: BashContainer, edit_container: EditContainer):
        self.edit_container = edit_container
        self.bash_container = bash_container
        
    async def pdf_to_markdown(
        self, 
        url: Annotated[str, Field(
            description="The url of the pdf to convert to markdown"
        )], 
        page_range: Annotated[str | None, Field(
            description=(
            "The page range to convert to markdown. None means convert the entire pdf, else "
            "Page range to convert, specify comma separated page numbers or ranges. Example: 0,5-10,20 will convert pages 0, 5-10, and 20"
        ))],
        force_redo: Annotated[bool, Field(
            description="Whether to force redoing the conversion instead of using the cached version"
        )],
    ) -> str:
        base_dir = await self.bash_container.ensure_cwd()
        
        if isinstance(self.bash_container._session, _ModalBashSession):
            save_remote = True
        else:
            save_remote = False
        
        with modal.enable_output():
            with app.run():
                o = pdf_to_markdown.remote(url, base_dir, page_range, save_remote, force_redo)
                if not save_remote:
                    data_dict, title = o
                    path = save_pdf_to_markdown(base_dir, data_dict, title)
                    return path
                else:
                    return o
        

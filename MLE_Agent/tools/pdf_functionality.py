# we seperate the file from .pdf so we dont have to mount additional files like edit.py or bash.py to the sandboximport modal
from modal import Image, App, Volume
import os
import re
import time
import subprocess
from pathlib import Path

agent_volume = Volume.from_name("mle-sandbox", create_if_missing=True)
# Use a dedicated Modal app for PDF functionality to avoid conflicts with
# any concurrently running app (e.g., the Modal Sandbox in mcp_server).
# Defining a fresh App here also ensures proper hydration for @app.function
# without requiring a prior lookup/deploy of an existing app.
app = App("mle-agent-pdf")

def run_marker_install():
    # hacky way to install marker models as part of the image 
    # by running on dummy arxiv paper
    import subprocess
    arxiv_link = "https://arxiv.org/pdf/2002.00719"
    marker_path = '/root/data/marker'
    os.makedirs(marker_path, exist_ok=True)
    subprocess.run(["curl", arxiv_link, '-o', marker_path + '/input.pdf'], check=True)
    subprocess.run([
        "marker", marker_path, 
        '--output_format', 'markdown', 
        '--page_range', '1', 
        '--output_dir', marker_path + '/output'
    ], check=True)

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

# Mount the local repo so the remote runtime can import the 'tools' package.
_REPO_ROOT = Path(__file__).resolve().parents[1]

image = Image.debian_slim(python_version="3.12").apt_install(
    "git",
    "curl",
).pip_install(  # required to build flash-attn
    "ninja",
    "packaging",
    "wheel",
).pip_install(
    "uv",
).run_commands(
    "uv pip install --system torch==2.8.0",
    "uv pip install --system 'transformers>=4.51.2,<4.54.0'",
    "uv pip install --system marker-pdf==1.9.0",
    "uv pip install --system datasets>=3.4.1",
    "uv pip install --system einops>=0.8.1",
    "uv pip install --system psutil>=6.1.1",
    "uv pip install --system hf-transfer",
).env(
    {
        "VIRTUAL_ENV": "/usr/local",
        "PATH" : "/usr/local/bin:$PATH",
        "HF_HUB_ENABLE_HF_TRANSFER" : "1",
        "GIT_USER_NAME": os.environ.get("GIT_USER_NAME", "Default Name"),
        "GIT_USER_EMAIL": os.environ.get("GIT_USER_EMAIL", "default@example.com")
    }
).run_commands(
    "git config --global user.name \"$GIT_USER_NAME\"",
    "git config --global user.email \"$GIT_USER_EMAIL\""
).run_function(run_marker_install).run_commands(
    'uv pip install --system modal'
).add_local_dir(
    str(_REPO_ROOT),
    "/root",
    copy=True
)


def split_markdown_into_pages(base_filename: str) -> list[str]:
    """
    Split markdown content into per-page files with naming convention file_path.name_page_{page_number}
    
    Args:
        markdown_content: The full markdown content with page separators
        base_filename: The base filename without extension
        output_dir: Directory to write the page files
    
    Returns:
        List of file paths for the created page files
    """
    import os
    assert os.path.exists(base_filename), f"File {base_filename} does not exist"
    assert base_filename.endswith('.md'), f"File {base_filename} is not a markdown file"
    with open(base_filename, 'r') as f:
        markdown_content = f.read()
        
    # Pattern to match page separators: {page_number}---PAGE---
    page_pattern = r'\{(\d+)\}---PAGE---\n(.*?)(?=\{\d+\}---PAGE---|$)'
    
    # Find all page matches
    page_matches = re.findall(page_pattern, markdown_content, re.DOTALL)
    
    created_files = []
    
    for page_num, page_content in page_matches:
        # Clean up the page content (remove leading/trailing whitespace)
        page_content = page_content.strip()
        
        # Create filename with the specified conventio
        dir = os.path.dirname(base_filename)
        name = os.path.basename(base_filename)
        page_filename = f"{name.split('.')[0]}_page_{page_num}.md"
        print(f"Page filename: {page_filename}")
        page_filepath = os.path.join(dir, page_filename)
        
        # Write the page content to file
        with open(page_filepath, 'w', encoding='utf-8') as f:
            f.write(page_content)
        
        created_files.append(page_filepath)
        print(f"Created page file: {page_filepath}")
    
    return created_files

def _sanitize_title(title: str) -> str:
    # Lowercase, replace spaces with dashes, remove invalid path chars
    t = title.strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"[^a-z0-9\-\.]+", "", t)
    return t[:120] or "paper"

def _parse_arxiv_info(url: str) -> tuple[str, str | None, str | None]:
    """Return (pdf_url, title, arxiv_id) if url is arXiv; else (url, None, None)."""
    import subprocess
    import re as _re

    def _fetch(url_: str) -> str:
        res = subprocess.run(["curl", "-sL", url_], check=True, capture_output=True)
        return res.stdout.decode("utf-8", errors="ignore")

    m_abs = _re.search(r"arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5}(v\d+)?)", url)
    if not m_abs:
        # New-style IDs with subject classes are also supported by arXiv, attempt another pattern
        m_abs = _re.search(r"arxiv\.org/(abs|pdf)/([a-z\-]+/\d{7}(v\d+)?)", url)
    if m_abs:
        arxiv_id = m_abs.group(2)
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url_ = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        html = _fetch(abs_url)
        # Prefer citation_title meta
        title = None
        mt = _re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', html, _re.I)
        if mt:
            title = mt.group(1).strip()
        if not title:
            # Fallback to <title>... - arXiv
            tt = _re.search(r"<title>(.*?)</title>", html, _re.I | _re.S)
            if tt:
                title = tt.group(1).split("- arXiv")[0].strip()
            
        return pdf_url_, title, arxiv_id
    return url, None, None

def _guess_title_from_url(url: str) -> str | None:
    # Try Content-Disposition filename or final path segment
    try:
        head = subprocess.run(["curl", "-sI", url], check=True, capture_output=True)
        headers = head.stdout.decode("utf-8", errors="ignore")
        m = re.search(r"filename=\"?([^\";]+)\"?", headers, re.I)
        if m:
            name = os.path.splitext(os.path.basename(m.group(1).strip()))[0]
            return name
    except Exception:
        pass
    # Fallback to URL path
    try:
        name = os.path.splitext(os.path.basename(url.split("?", 1)[0]))[0]
        # Replace separators with spaces for readability before sanitizing later
        return re.sub(r"[_\-]+", " ", name).strip() or None
    except Exception:
        return None
    

def save_pdf_to_markdown(
    base_dir: str,
    dict_bytes: dict[str, bytes], 
    title: str | None = None
) -> str:
    
    title = title or "paper"
    title = _sanitize_title(title)
    # Extract metadata for naming


    root = Path(base_dir)
    dest_dir = root / "papers" / title
    
    os.makedirs(dest_dir, exist_ok=True)
    markdown_file = None
    for name, value in dict_bytes.items():
        out_path = dest_dir / name
        
        os.makedirs(out_path.parent, exist_ok=True)
        print(f"Writing file {out_path} with value {value.decode('utf-8')}")
        # dont write but use the edit container to write
        with open(out_path, "w") as f:
            f.write(value.decode("utf-8"))
        if name.endswith(".md"):
            markdown_file = out_path

    # Optionally split into per-page files using EditContainer abstraction
    if markdown_file:
        try:
            split_markdown_into_pages(str(markdown_file))
        except Exception as e:
            print(f"Warning: failed to split markdown via edit container: {e}")

    if markdown_file:
        return str(markdown_file)
    else:
        return str(dest_dir.absolute())


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 10,
    serialized=True,
    volumes={
        "/root/sandbox": agent_volume,
    }
)
def pdf_to_markdown(
    pdf_url: str, 
    base_dir: str, 
    page_range: str | None = None,
    save_remote : bool = True    
) -> str | tuple[dict[str, bytes], str]:
    """
    Converts a PDF file to Markdown and images using pdf-marker.
    Returns a dictionary mapping relative file paths to their binary content.
    """

    import subprocess
    import os
    import tempfile
    import torch
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        output_dir = temp_dir / "output"
        pdf_path = temp_dir / "input.pdf"
  
        # Normalize URL and attempt to fetch title if arXiv
        arxiv_pdf_url, arxiv_title, arxiv_id = _parse_arxiv_info(pdf_url)
        source_url = pdf_url
        pdf_url = arxiv_pdf_url

        # Download PDF
        subprocess.run(["curl", "-L", pdf_url, "-o", pdf_path], check=True)
        
        print(f"PDF path: {pdf_path}")
        print(f"Temp dir: {temp_dir} {os.listdir(temp_dir)}")
        
        assert torch.cuda.is_available(), "CUDA is not available"
        t0 = time.time()
        cmds = [
            "marker_single", pdf_path, 
            "--output_format", "markdown",
            "--paginate_output", "--MarkdownRenderer_page_separator", '---PAGE---',
            "--output_dir", output_dir
        ]
        if page_range:
            cmds.extend(["--page_range", page_range])
        
        subprocess.run(cmds, check=True)
        print(f"Marker took {time.time() - t0} seconds")
        print(f"Markdown files saved to {output_dir} {os.listdir(output_dir)}")
        # Collect all files in the output directory
        result = {}
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path != pdf_path:
                # Use relative path from temp_dir as key
                with open(file_path, 'rb') as f:
                    result[file_path.name] = f.read()

        # Build and attach metadata to aid client naming
        md_guess = None
        for k in list(result.keys()):
            if k.endswith(".md"):
                md_guess = k
                break
            
        title_guess = (
            arxiv_title or 
            _guess_title_from_url(source_url) or 
            (md_guess or "paper").replace(".md", "")
        )
        
        print(f"os.listdir({base_dir}): {os.listdir(base_dir)}")
        
        if save_remote:
            return save_pdf_to_markdown(base_dir, result, title_guess)
        else:
            # if we transmit over the network to our client and save there
            return result, title_guess
        
        

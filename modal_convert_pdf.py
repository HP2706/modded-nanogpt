from pathlib import Path
from modal import Image, gpu, Secret, App, Volume
import os
import time
import re

volume = Volume.from_name("mle-agent-volume", create_if_missing=True)
app = App(name="nanogpt")


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
).run_function(run_marker_install)


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


@app.function(
    image=image, 
    gpu="A10G", 
    timeout=60 * 10,
)
def pdf_to_markdown(pdf_url: str, page_range: str | None = None) -> dict[str, bytes]:
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
        subprocess.run(["curl", pdf_url, '-o', pdf_path], check=True)
        
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
        
        print("file paths", result.keys())
        print(f"Result: {result}")
        return result
    
def pdf_tool(url: str, paper_name_or_path: str, page_range: str | None = None) -> str:
    '''
    Given url of a pdf, return a string path to the pdf with all the images and markdown.
    Also splits the markdown into per-page files with naming convention file_path.name_page_{page_number}
    
    Args:
        url: url of the pdf
        paper_name_or_path: name of the paper
        page_range: range of pages to convert
        
    Returns:
        string path to the pdf with all the images and markdown
    
    Raises:
        AssertionError: if the pdf does not exist or is not a markdown file
    '''
    assert '/pdf/' in url, "URL must point to a pdf and contain /pdf/"
    dict_bytes = pdf_to_markdown.remote(url, page_range)
    import os
    print(f"Dict bytes: {dict_bytes.keys()}")
    
    # Find the markdown file in the dictionary
    markdown_file = None
    markdown_content = None
    for path, value in dict_bytes.items():
        file_path = f"{paper_name_or_path}/{path}"
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if path.endswith('.md'):
            markdown_file = path
        
        with open(file_path, 'wb') as f:
            f.write(value)
    
    
    if markdown_file:
        return f"{paper_name_or_path}/{markdown_file}"
    else:
        # Fallback if no markdown file found
        return f"{paper_name_or_path}/"
    


@app.local_entrypoint()
def main(
    url: str,
    paper_name: str,
    page_range: str | None = None,
):
    path = pdf_tool(url, f"papers/{paper_name}", page_range)
    print(f"Markdown generated at {path}")

if __name__ == "__main__":
    main(url="https://arxiv.org/pdf/2505.14669", paper_name="4-bit", page_range="1-4")
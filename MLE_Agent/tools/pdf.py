from modal import App
import modal
from .pdf_functionality import pdf_to_markdown, save_pdf_to_markdown, app
from .bash import BashContainer
from .edit import EditContainer



class PdfContainer:
    def __init__(self, bash_container: BashContainer, edit_container: EditContainer):
        self.edit_container = edit_container
        self.bash_container = bash_container
        
    async def pdf_to_markdown(
        self, 
        url: str, 
        page_range: str | None = None
    ) -> str:
        base_dir = await self.bash_container.ensure_cwd()
        
        if self.bash_container._modal_session is not None:
            save_remote = True
        else:
            save_remote = False
        
        with modal.enable_output():
            with app.run():
                o = pdf_to_markdown.remote(url, base_dir, page_range, save_remote)
                print(f"o: {o}")
                if not save_remote:
                    data_dict, title = o
                    path = save_pdf_to_markdown(base_dir, data_dict, title)
                    return path
                else:
                    return o
        


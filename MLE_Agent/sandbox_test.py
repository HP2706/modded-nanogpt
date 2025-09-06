from modal import Sandbox, Image
import modal
from modal.stream_type import StreamType

def test_sandbox():
    try:
        app = modal.App.lookup("my-app", create_if_missing=True)
        with modal.enable_output():
            
            image = Image.debian_slim(python_version="3.12")

            
            image = image
            sb = Sandbox.create(
                app=app, 
                image=image,
                volumes={
                    '/root/data' : modal.Volume.from_name("gpt2-speedrun", create_if_missing=True)
                },
                timeout=30,
                verbose=True
            )


       
        p = sb.exec(
            "bash", "-c",
            "cd /root/sandbox && ls . && python test.py",
            timeout=30,
            stderr=StreamType.STDOUT,
            stdout=StreamType.STDOUT
        )
        p.wait()
       
        sb.terminate()
    finally:
        sb.terminate()
    

if __name__ == "__main__":
    test_sandbox()
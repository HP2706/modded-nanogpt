import modal

app = modal.App(name="recursive-modal")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install('modal')

@app.function(
    image=image,
)
def recursive_modal():
    
    inner_app = modal.App(name="inner-modal")
    
    @inner_app.function(
        serialized=True
    )
    def inner_modal():
        return "Hello, world! from inner modal"
    with inner_app.run():
        inner_msg = inner_modal.remote()
    
    print("inner_msg", inner_msg)
    print("Hello, world! from outer modal")
    
    

@app.local_entrypoint()
def main(): 
    recursive_modal.remote()
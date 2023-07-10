import modal


stub = modal.Stub()
custom_image = modal.Image.from_dockerfile("Dockerfile",
                                           context_mount=modal.Mount.from_local_dir("./", remote_path="/"))

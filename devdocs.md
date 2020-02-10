### Pushing new mybinder image

1) Commit all changes to be included in image.


2)
    ```bash
    TAG=$(git rev-parse --short HEAD) dc build
    TAG=$(git rev-parse --short HEAD) dc push
    ```
    
3) Change `Dockerfile` to point to exact commit given by `git rev-parse --short HEAD`.

### Pushing new documentation

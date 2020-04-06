### Documentation

To write new documentation:

1. Use Jupytext to edit or create new notebooks in Markdown format in `docs/src`. Note: Jupytext makes it so that only the input cells are saved to the repository. 

2. You can verify locally that the documentation builds by running `docker-compose build` then `docker-compose up`. If it works here, its guaranteed to work when Github Actions builds the documentation from the notebooks and when the notebooks are launched by MyBinder, since this identical Docker container is used in both cases. 

3. Once the documentation is verified, commit changes to master or submit a PR. If you submit a PR, Github Actions will build a preview of the documentation for you, so you can check that rather than verifying locally if you like. 

The Github Actions workflow will build documentation for master and for all tags and PRs. It pushes these to the `gh-pages` branch of CMBLensing.jl which is then hosted at [cosmicmar.com/CMBLensing.jl](https://cosmicmar.com/CMBLensing.jl)

All MyBinder links point to the `gh-pages` branch of CMBLensing.jl. MyBinder uses the `Dockerfile` in the root folder of this branch which is a symlink pointing to `stable/Dockerfile`, and `stable/` is itself a symlink managed by Documenter.jl made to point at the most recent tagged version. This means MyBinder links from any version of the documentation will always point to the most recent stable tagged vesion (ideally they would point to whatever version the documenation itself was, but that's harder to set up).

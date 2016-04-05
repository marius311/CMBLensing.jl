# BayesLensSPTpol

[![Build Status](https://travis-ci.org/EthanAnderes/BayesLensSPTpol.jl.svg?branch=master)](https://travis-ci.org/EthanAnderes/BayesLensSPTpol.jl)


# Hi Zhen and Lloyd.

I've started a github repo for all the future Julia code we will write for the Bayesian CMB polarization lensing project we are thinking about. This repo is currently in "private mode" so we are the only ones who can see it. If the work eventually becomes publishable, we can switch to "public mode" and add instructions on how to reproduce all the results in the paper. I've actually organized it as a Julia package to make it easy for others to eventually load in the library and run the result.

# Using git for collaboration

Here are a few tips on how to use git for collaboration in small groups.

## Downloading and running the code

I've assigned you both as collaborators to this repo so it should already be accessable in your github account. To download it to your computer, use the following command in the terminal (in the directory which you want the folder downloaded to)

```
$ git clone https://github.com/EthanAnderes/BayesLensSPTpol.git
```

Now, to make sure Julia can find the library, add the following command to your Julia startup file `~/.juliarc.jl`

```julia
push!(LOAD_PATH, "<path to the directory containing BayesLensSPTpol directory>")
```

If you don't want to add this to your Julia startup file, you will need to run the above command at the Julia command line each time you start up Julia and want to run this code.

Now you should be able to run the following commands from within Julia.

```julia
julia> using BayesLensSPTpol   # loads the package
julia> cd("<path to BayesLensSPTpol/figures/>")
julia> include("make_figure_1.jl")
```


## Workflow for using git and Github.


I use a basic workflow for using git in a collaboration. It seems to work well. The basic idea is this: there is a version of the code called `master`. This version holds the most up to date version of the code and the paper. When you sit down to make some changes, you need to create a `branch`, which copies the current `master` into something you can edit without initially changing master. Now you can edit what ever you want on your `branch`. When your done editing just switch to `master`, update `master` to incorporate any changes that have been made since you created your `branch`, merge your `branch` into `master`, then upload the new `master` to github and delete the `branch` you just merged.

The nice thing about this way of doing things is that we can all work separate branches simultaneously and then merge them when done. Sometimes there are conflicts, but these are usually pretty easy to handle.
This also means that I can create a `branch` just to mess around and brake things, deleting the `branch` without merging to `master` and wrecking things. Think about a `branch` as lasting anywhere from 1 day to a couple weeks.

Here are the steps for this workflow.


### Step 1: Git pull any new changes to master

Check the status of your local git repo by navigating to the `BayesLenseSPTpol` directory and typing the following command at the terminal.  
```
git status
```
At this point you want to make sure your on the `master` branch (and the above command will tell you which branch your on).


Check that there are no other branches from the previous days work.
```
git branch
```


Check the remote branches (all the branches that others have pushed to Github but that aren't merged with master yet) to see what others are currently working on.
```
git branch -r
```


Now load any new remote changes to `master`.
```
git pull
```


You can also see a graph of the commits.
```
git log --graph --full-history --all --color --pretty=format:"%x1b[31m%h%x09%x1b[32m%d%x1b[0m%x20%s"
```




### Step 2: Create a new branch, commit to it, and push to Github so others can see it.
==========================================================

Create a new local branch.
```
git branch EA/debug_lensedBmode
git checkout EA/debug_lensedBmode
```


Locally make commits.
```
git add --all
git commit -m "commit message"
```


Push the commits up to github.
```
git push origin EA/debug_lensedBmode
```


If, at any time, you want to switch to another branch.
```
git checkout another_branch
```




### Merger to master when done.

When finished with branch `EA/debug_lensedBmode` now we want to merge with with local master and update the origin master accordingly.


First switch to master and fetch any recent updates
```
git checkout master
git pull
```


Now merge the branch `EA/debug_lensedBmode` into `master`
```
git merge EA/debug_lensedBmode
```


Now push the locally merged `master` up to the remote `master` on Github.
```
git push origin master
```





### Deleting your old branch

Locally delete EA/debug_lensedBmode after merging
```
git branch  -d EA/debug_lensedBmode
```


Now delete the branch on github
```
git push origin :EA/debug_lensedBmode
```

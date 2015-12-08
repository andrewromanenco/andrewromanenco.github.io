---
layout: post
title:  "Webapp as a GIT server: hello world"
date:   2015-12-08
tags: [git, ruby]
---

At the end of this exercise, a web application will be up and running. Executing something like ```git clone http://localhost:4000/ hello-world-repo``` will clone a git repo with single text file in it. Yes, clone from a web application with no git on server side.

What is covered:

 - git objects formats
 - git objects creation
 - endpoints to support client requests
 - dumb protocol

What is not covered:

 - packs (the way how git optimizes storage)
 - smart protocol

[The app is available @github](https://github.com/andrewromanenco/git-server-hello-world)

# What is a git repo

On a high level, a git repo is a set of commits. Each commit has optional link to a parent one; and required link to a tree structure. This tree structure is just a list of links to files and other trees. As simple as that.

On a low level, git uses key-value storage to keep all objects. These keys are hash values for the content and these hashes are used as links in references.

To have a repo with single file in it; these steps must be completed:

 - calculate a hash value (H1) for some test content
 - create a tree object with a file name and a link (H1) to the content; and calculate a hash for the tree (H2)
 - create a commit object with a link (H2) to the tree object; and calculate a hash for the commit itself (H3)

Because hashes have dependencies H3 &lt;- H2 &lt;- H1, any change to the content, the file name or the commit will change some or all hashes as well. Another conclusion is that to calculate a hash for the commit, hashes for the tree and the content must already be calculated.

# Implementation

The sample code is a simple rails application. Turning this app into a demo git repository may be splitter into two steps:

 - calculate hashes for the content, the tree and the commit; and caching them
 - handle git requests over the HTTP dumb protocol and using a cache to transfer data to a client during a cloning operation

[Entire code lives in single controller @github](https://github.com/andrewromanenco/git-server-hello-world/blob/master/app/controllers/git_controller.rb). It should be quite easy to reimplement the logic in any language.

## Calculate hashes & cache

### 0. Git transfers archived data.

To keep things simple, our cache is going to contain archived data as well; it will simplify request/response calls for git clone command.

### 1. Content (build_file method in the source)
Git stores content as blob objects. The format is ```blob SIZE\0content```. \0 is zero byte and SIZE = size_in_bytes(content).
Hash get calculated over this data(blob keyword, size and original content). After the calculation is done; the data is archived and stored to the cache (simple hash map).

### 2. Tree (build_tree)
Tree is a listing of files and other trees; and is, basically, a list of items. Each item has format: ```file_permissions file_name\0file_hash```. There are few things to keep in mind. First of all, file_hash is the hash calculated on step 1; it’s added to the item as 20 bytes (and not as 40 bytes HEX representation). Secondly, there are no separator between items (no spaces or new lines).

With a list of items in a variable, tree record can be created according to this format: ```tree SIZE\0list_of_items```; where SIZE is size_in_bytes(list_of_items). As usual, hash get calculated over this object and everything is cached (after archiving).

### 3. Commit(build_commit)
Format: ```tree TREE_HASH\n\nCOMMIT_MESSAGE```. And the object is ```commit SIZE\0CONTENT```. Note, TREE_HASH for the commit is a HEX representation. In case parent commit should be referenced, ```parent HASH_OF_OTHER_COMMIT\n``` goes before the tree link. Commit is cached and archived as usual.

## Handle git clone

As it was told before, on a low level git is key-value storage. Basically, the application exposes an endpoint - ```object/*hash```, where a client can retrieve an object by providing a hash (as a lowercase hex string). (See routes file for details.)[https://github.com/andrewromanenco/git-server-hello-world/blob/master/config/routes.rb] The handler for this endpoint (method objects) simply looks into the cache to return the data to a caller.

This demo keeps all objects in the cache, so the ```object/*hash``` endpoint always returns a result. In a full git implementation, this endpoint could return 404(not found), which means that the object is stored in a pack file. Pack files are outside of the scope for this app; please, read git documentation for details.

With objects available by their hashes, there are two question two answer: how does git knows hash values to ask for; and how git knows which git branch to checkout locally after clone.

Second question is simple: there is an endpoint ```heads```, which returns a name for the default branch. In this example app, the name is hardcoded to 'ref: refs/heads/master\n’ - see head method; so master get checked out after a clone.

To answer first question there is another endpoint: ```info/refs```. It lists all available heads (each head is a commit). In this demo, there is only one head/commit and a single record is returned: ```COMMIT_HASH\trefs/heads/master\n```. See info_refs method for details.

# Clone it

With the app up and running, execute: ```git clone http://localhost:3000 hello-world```

To sum it up; these are steps taken by git clone command against the demo app:

 1. Request a list of available heads (single commit is available)
 2. Request head name to be checked out after clone is done (always master)
 3. Use hash to retrieve an object
 4. Parse the object; and in case it’s a commit or a tree,  find all new hashes inside, and resolve each one of them starting from step 3
 5. When no more new hashes are available, checkout a head, identified on step 2
 6. Tell user that the cloning is done!

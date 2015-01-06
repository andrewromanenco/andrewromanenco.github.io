---
layout: post
title:  "Dynamic Programming"
date:   2015-01-05
tags: [dynamic-programming]
---

<a href="{{ site.baseurl }}/tag/dynamic-programming/" target="_blank">These posts</a> show how to solve problems with dynamic programming step-by-step.

Each post follows a plan:

 - build brute force recursive solution
 - get surprised how slow it is and figure out why
 - improve solution with memoization
 - convert to “true” dynamic-programming bottom-up solution

I assume that you already know what dynamic programming is; here are few links:

 - <a href="http://en.wikipedia.org/wiki/Dynamic_programming" target="_blank">Dynamic Programming @ Wikipedia</a>
 - There is a good tutorial <a href="http://www.topcoder.com/tc?d1=tutorials&d2=dynProg&module=Static" target="_blank">@topcoder</a>
 - And very nice video explanation problem-by-problem <a href="http://people.cs.clemson.edu/~bcdean/dp_practice/" target="_blank">here</a>

Still, many people look for “better” way to approach DP problems. This is what these posts are about.

### Some theory

Basically, DP is an opportunity to solve a problem of two characteristics: overlapping subproblems and optimal substructure. With this properties in mind we can exchange memory-to-time: run faster with greater memory use.

<a href="http://en.wikipedia.org/wiki/Overlapping_subproblems" target="_blank">http://en.wikipedia.org/wiki/Overlapping_subproblems</a><br/>
<a href="http://en.wikipedia.org/wiki/Optimal_substructure" target="_blank">http://en.wikipedia.org/wiki/Optimal_substructure</a>

To keep definition short:

Overlapping subproblems: your solution keeps solving same sub-problems on and on.

Optimal substructure: on each step your solution is based on optimal solutions of already solved subproblems.

Top-down approach: you solve bigger problem, by calling the same logic on smaller chunks of input - this is recursive solution.

Memoization: save results of already calculated calls to smaller subproblems and reuse these results instead of recalculating the same numbers on and on.

Bottom-up approach: small problems are solved first, and then combined to more generic ones, until we get to the goal.

<a href="{{ site.baseurl }}/tag/dynamic-programming/" target="_blank">click here for the list of posts</a>

<a href="https://github.com/andrewromanenco/dynamic-programming/" target="_blank">source code @gitub</a>
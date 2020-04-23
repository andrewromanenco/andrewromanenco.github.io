---
layout: post
title:  "Five levels of Leetcode (prep for a coding interview)"
date:   2020-04-22
tags: [software-development]
---

Coding interviews are a pretty common step in joining a technology company. Candidates are asked to solve coding problems during both phone interviews and in on-site ones. Many on-site loops have several coding exercises included.

Every successful coding interview is based on two factors: knowledge and practice. Knowledge part covers main data structures and algorithms. In addition, you should know very well the programming language of your choice. Any mainstream language works. The second contributing factor to the success of an interview is practice. Interview is an environment with time constraints and, maybe, some stress involved. Luckily, practice makes things much easier.

I would like to focus on the second piece of a successful coding interview, the practice. There are many ways you could approach this step. For example, you could write a solution for a problem, and then write unit tests. In the long run, this is a great approach. Writing tests by hand forces the thinking process to consider edge cases.

Writing tests by hands is a good idea. The problem is the quality. There is no easy way to check if all edge cases are covered. A simplification for the testing step is using some platform where tests are provided to you, and all you need is to write a solution. Writing a solution for a well written test will force you to cover edge cases in your code.

Online judges services allow submitting a solution to a problem and the solution is executed against a predefined test set. Many of these judges exist. The one I would like to recommend is [Leetcode](https://www.leetcode.com). The platform has hundreds of problems and supports many programming languages. The name “leetcode” became a common noun for practicing coding interview questions; e.g. there are many questions online like “How many weeks do I need to leetcode to get to company X?”.

I am using [Leetcode](https://www.leetcode.com) for both interview preparation and for keeping my knowledge sharp. Also, leetcode is fun, for a code junkie.

Due to a wide range of available problems and a great community, Leetcode helps to polish coding skills on many levels. I would like to call out these levels.

Level 1: brute force

You should be able to solve (almost) any problem with a brute force algorithm. Brute force requires ability to write down a thought process in code. A surprisingly large number of candidates fail on this level. A candidate may say “I will sort data points and pick the largest element. Then I’ll repeat this process again”, and then not be able to actually write it down. This indicates that there is not enough practice.

Leetcode will accept your brute force solution for execution and will tell you if either memory or time is not good. At the same time, some basic test cases will pass. Time to go to level 2.

Level 2: optimization

A better approach may exist, which will improve one or more aspects of the solution. Usually, the improvement goes to either memory or runtime. It is up to you to decide which one to go with. In an interview, this is a great discussion point.

Sometimes, the optimization results in a completely new solution, and it is better in both runtime and memory. This solution may require new data structures and/or algorithms.

All tests must pass on this level.

Level 3: alternatives

The level 2 solution is good. It is both fast and does not use too much memory. What are alternative solutions? For example, some graph based problems may be solved with union-find algorithms. The alternative solutions may be relatively faster or slower. The important point is to find them. Coding them through gives an extra practice bonus.

Strictly speaking, finding alternatives for the brute force step also makes sense. Coming up with a list of worst possible solutions is lots of fun and really tests base knowledge.

Level 4: code quality

Level 3 code works and is within set limits. Does the source code itself have a good quality? Is it easy to understand? Would it get any red flags during a code review? Having the code clean is as important as having the code correct.

The community part of Leetcode can help. Every problem on Leetcode has a forum, with other engineers sharing their solutions. These solutions will help to evaluate your performance in both levels 3 and 4. If you missed an approach, it will be in the forum. If your solution could be written in a more elegant way, an example will be available.

Level 5: tricks

Some problems have nice tricky solutions. These approaches are not the best ones, as they are hard to understand. But there is nothing wrong in being aware about them. The forum will help to find those.

Have a work log

It is important to keep a track of all issues happening while working on a problem. Have a written log of all small things: failure of compilation - write down why. Missed a solution? Write down a few sentences about why you didn’t see it. The chances are you have strong and weak areas in your knowledge. A detailed log helps to analyze mistakes and helps to find items to improve. Knowing about a weak spot is a first step to make it strong.

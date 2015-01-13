---
layout: post
title:  "Unique paths problem"
date:   2015-01-06
tags: [dynamic-programming]
---

Read <a href="{{ site.baseurl }}/dynamic-programming-intro/" target="_blank">Dynamic Programming intro</a> first.

`A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. How many possible unique paths are there?`

Source code for this post available <a href="https://github.com/andrewromanenco/dynamic-programming/tree/master/src/main/java/com/romanenco/dp/uniquepaths/" target="_blank">@github</a> as maven based java project.

### Our plan

 - build brute force recursive solution
 - get surprised how slow it is and figure out why
 - improve solution with memoization
 - convert to “true” dynamic-programming bottom-up solution

### Brute force solution

After drawing several sample inputs (for example 2x2 and 2x3) we can notice a dependency: for every given cell, number of ways to move to bottom-right corner of the grid is sum of paths from other two cells, one to the right and one to the bottom. This gives us an idea of a recursive solution.

{% highlight java %}
public class UniquePathsBF {

    public long uniquePathsTD(int m, int n) {
        return paths(0, 0, m, n);
    }

    private long paths(int i, int j, int m, int n) {
        if ((i == m)||(j == n)) {
            return 0;
        }
        if ((i == m - 1)||(j == n - 1)) {
            return 1;
        }
        return paths(i + 1, j, m, n) + paths(i, j + 1, m, n);
    }

}
{% endhighlight %}

Here are running times for unit tests:

 - testSmallInput() - 2x3 - 0.0 seconds
 - testLargeInput() - 15x21 - 4.7 seconds
 - testLargerInput() - 16x21 - 11.3 seconds

The code gets slower and slower with every increment of an input. After trying to debug what's going on for some small input, it's easy to see that the same cells get recalculated on and on. Basically, we see overlapping sub-problems here.

### Top-down with memoization

The idea from above gives us an easy opportunity to exchange memory-to-time by caching already calculating results. Obviously, our cache is going to be of size similar to the grid.

{% highlight java %}
public class UniquePathsTD {

    public long uniquePathsTD(int m, int n) {
        final long[][] memo = new long[m][n];
        return paths(0, 0, m, n, memo);
    }

    private long paths(int i, int j, int m, int n, long[][] memo) {
        if ((i == m)||(j == n)) {
            return 0;
        }
        if (memo[i][j] > 0) {
            return memo[i][j];
        }
        if ((i == m - 1)||(j == n - 1)) {
            return 1;
        }
        final long result = paths(i + 1, j, m, n, memo)
                + paths(i, j + 1, m, n, memo);
        memo[i][j] = result;
        return result;
    }

}
{% endhighlight %}

This implementation is not suffering from repetitive recalculations and works pretty fast.

### Bottom-up solution

Now, we can trace how a cache gets populated with data; and reverse the flow for a bottom-up solution without any recursive calls.

Define array paths[m][n] where each element contains number of ways how one can reach it from top-left cornet. Base case is there, for every first row and first column the result is always one. And every next cell is sum of paths[i - 1][j] and paths[i][j - 1]. Our final number is available in paths[m - 1][n - 1].

{% highlight java %}
public class UniquePathsBU {

    public long uniquePathsTD(int m, int n) {
        final long[][] paths = new long[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            paths[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            paths[0][j] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                paths[i][j] = paths[i - 1][j] + paths[i][j - 1];
            }
        }
        return paths[m - 1][n - 1];
    }

}
{% endhighlight %}

Basically, we are using optimal substructure to calculate next step as a combination of smaller tasks.

Bottom line: build brute force recursive solution -&gt; get frustrated with speed -&gt; add memoization -&gt; convert to bottom-up solution by looking into recursion from opposite way.
---
layout: post
title:  "Minimum number of jumps problem"
date:   2015-01-28
tags: [dynamic-programming]
---
Read <a href="{{ site.baseurl }}/dynamic-programming-intro/" target="_blank">Dynamic Programming intro</a> first.

**Please, read till the end, as this problem has a solution better than dp**

`Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Your goal is to reach the last index in the minimum number of jumps.`

Source code for this post available <a href="https://github.com/andrewromanenco/dynamic-programming/tree/master/src/main/java/com/romanenco/dp/jumps/" target="_blank">@github</a> as maven based java project.

### Our plan

 - build brute force recursive solution
 - get surprised how slow it is and figure out why
 - improve solution with memoization
 - convert to “true” dynamic-programming bottom-up solution
 - bonus step

### Brute force solution

Recursion is a good starting point for brute force solution. If we are in position i, we know which positions to the right are reachable with one additional jump. Same rule applies to those right positions as well.

{% highlight java %}
public class MinJumpsBF {

    public int jump(int[] A) {
        if (A.length < 2) {
            return 0;
        }
        return jump(A, 0);
    }

    private int jump(int[] A, int index) {
        if (index >= A.length - 1) {
            return 0;
        }
        int min = Integer.MAX_VALUE;
        for (int i = 1; i <= A[index]; i++) {
            min = Math.min(min, 1 + jump(A, index + i));
        }
        return min;
    }

}
{% endhighlight %}

Running the code on small input works, but with longer array there is significant delay. After tracing the code, it's easy to see that we call jump(array, index) many times for the same value of index. Caching should help us.

### Top-down with memoization

{% highlight java %}
public class MinJumpsTD {

    public int jump(int[] A) {
        if (A.length < 2) {
            return 0;
        }
        final int[] memo = new int[A.length];
        Arrays.fill(memo, -1);
        return jump(A, 0, memo);
    }

    private int jump(int[] A, int index, int[] memo) {
        if (index >= A.length - 1) {
            return 0;
        }
        if (memo[index] != -1) {
            return memo[index];
        }
        int min = Integer.MAX_VALUE;
        for (int i = 1; i <= A[index]; i++) {
            min = Math.min(min, 1 + jump(A, index + i, memo));
        }
        memo[index] = min;
        return min;
    }

}
{% endhighlight %}

As usual, the only change we made is to have a cache and use that cache instead of recalculation information already known.

### Bottom-up solution

Tracing how the cache is filled in, we can replicate the process by bottom up approach. Last element of the cache is zero - this is our destination. Bottom up algorithm fills the cache in right-to-left direction, using data calculated earlier. And the result is in cache[0].

{% highlight java %}
public class MinJumpsBU {

    public int jump(int[] A) {
        if (A.length < 2) {
            return 0;
        }
        final int[] memo = new int[A.length];
        for (int i = A.length - 2; i >= 0; i--) {
            int min = Integer.MAX_VALUE;
            for (int k = 1; k <= A[i]; k++) {
                if (i + k < A.length) {
                    min = Math.min(min, 1 + memo[i + k]);
                }
            }
            memo[i] = min;
        }
        return memo[0];
    }

}
{% endhighlight %}

This solution works. But check out test cases. It's kind of slow for longer inputs. The reason is in O(n^2) running time for DP algorithm (you should make small change to the code from above to guarantee this estimate).

### DP is not always the best solution

DP is amazing method for solving specific problems. But the downside is that DP is an exchange of memory-to-time; and DP does not make any other optimizations.

This specific problem is solvable in linear time; by tracking longest reachable element and making smart decision when we increase number of steps.

{% highlight java %}
public class MinJumpsLinear {

    public int jump(int[] A) {
        if (A.length < 2) {
            return 0;
        }
        int steps = 0;
        int distance = 0;
        int update = 0;
        for (int i = 0; i < A.length - 1; i++) {
            if (i + A[i] > distance) {
                distance = i + A[i];
            }
            if (i == update) {
                steps++;
                update = distance;
                if (distance >= A.length - 1) {
                break;
            }
            }
        }
        return steps ;
    }

}
{% endhighlight %}

Bottom line: DP works well for this problem, comparing to brute force solution. But it always possible that more optimal approaches exist.
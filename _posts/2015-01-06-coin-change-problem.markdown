---
layout: post
title:  "Coin change problem"
date:   2015-01-06
tags: [dynamic-programming]
---

Read <a href="{{ site.baseurl }}/dynamic-programming-intro/" target="_blank">Dynamic Programming intro</a> first.

`You are given n types of coin denominations of values v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, so you can always make change for any amount of money C. Give an algorithm which makes change for an amount of money C with as few coins as possible.`

Source code for this post available <a href="https://github.com/andrewromanenco/dynamic-programming/tree/master/src/main/java/com/romanenco/dp/coinchange/" target="_blank">@github</a> as maven based java project.

### Our plan

 - build brute force recursive solution
 - get surprised how slow it is and figure out why
 - improve solution with memoization
 - convert to “true” dynamic-programming bottom-up solution

**Input:**<br/>
array of coin values and target sum C<br/>
**Output:**<br/>
min number of coins to use to sum up to C, we have indefinite supply for each coin denomination<br/>

For example: coins [1, 3, 4] and target 6. Answer is 2: 3+3. Note, that greedy approach would not work: 4 + 1+ 1, which is three coins.

### Brute force solution

Brute force solution is recursive. getMinNumberOfCoins has one base case: if target sum is zero, then we need zero coins to get it. Otherwise, we try to use each coin and ask the function again to get min number of coins for a smaller sum (current sum minus coin value).

{% highlight java %}
public class CoinChangeBF {

    public int getMinNumberOfCoins(int[] coins, int sum) {
        if (sum == 0) {
            return 0;  // base case
        }
        int result = Integer.MAX_VALUE;
        for (int coin: coins) {
            if (coin <= sum) {
                result = Math.min(
                        result,
                        getMinNumberOfCoins(coins, sum - coin) + 1
                        );
            }
        }
        return result;
    }

}
{% endhighlight %}



Let’s take a look to our <a href="https://github.com/andrewromanenco/dynamic-programming/blob/master/src/test/java/com/romanenco/dp/coinchange/CoinChangeBFTest.java" target="_blank">unit test</a>. If you want, run them one by one. test1 and test2 are fast (0.0 seconds); but what about testSlow and testSlower? See, they use same set of coins, but target sum is 40 and 41. My macbook's run time for these tests is:

 - test1() 0.0 seconds
 - test2() 0.0 seconds
 - testSlow() 2.7 seconds
 - testSlower() 4.9 seconds

There is HUGE difference for last two … and difference in target sum is just one. The reason is simple: this solution  has exponential execution time, relative to the value of the input. So if we will set target sum to a greater value - it will not get solved in any reasonable time.

What we gonna do?

Let’s try to see what’s going on inside … we know that all calls to our function are different only in target sum. So we will <a href="https://github.com/andrewromanenco/dynamic-programming/blob/master/src/main/java/com/romanenco/dp/coinchange/CoinChangeBFExtended.java" target="_blank">add simple map</a> to see how many times we’ve got called for every target amount...

 - 1: 74049690
 - 2: 45765225
 - 3: 28284465
 - 4: 17480761
 - 5: 10803704
 - 6: 6677056
 - 7: 4126648
 - ...

Ups, for the same input we call the function on and on … that's why it is SLOW!

With this knowledge we can optimize our original solution with very simple idea: we are going to cache results and before actually calling the function, we check, maybe the result is already there? This solution is top-down with memoization.

### Top-down with memoization

{% highlight java %}
public class CoinChangeTD {

    private final int[] coins;
    private final int sum;
    private final int[] memo;

    public CoinChangeTD(int[] coins, int sum) {
        this.coins = coins;
        this.sum = sum;
        memo = new int[sum + 1];
        for (int i = 0; i < sum + 1; i++) {
            memo[i] = -1;
        }
        memo[0] = 0;
    }

    public int getMinNumberOfCoins() {
        return getMinNumberOfCoins(sum);
    }

    private int getMinNumberOfCoins(int sum) {
        if (memo[sum] != -1) {  // eliminate duplicated calculations
            return memo[sum];
        }
        if (sum == 0) {
            return 0;  // base case
        }
        int result = Integer.MAX_VALUE;
        for (int coin: coins) {
            if (coin <= sum) {
                result = Math.min(
                        result,
                        getMinNumberOfCoins(sum - coin) + 1
                        );
            }
        }
        memo[sum] = result;  // save for reuse
        return result;
    }

}
{% endhighlight %}

Now, when we run our test case, the result is:

- test1() 0.0
- test2() 0.0
- testSlow() 0.0
- testSlower() 0.0

The solution is using cache to eliminate calculations solved earlier. The solution is faster, but has new memory requirements (memory-to-time exchange). Also note, the solution is now stateful.

### Bottom-up solution

Having code from above, it’s easy to see how the solution can be converted to bottom-up one: we want to calculate smaller problems first and use them as our building blocks for larger tasks.

Basically, we fill our memoization table from left to right. And our solution is the last element.

{% highlight java %}
public class CoinChangeBU {

    public int getMinNumberOfCoins(int[] coins, int sum) {
        int[] solution = new int[sum + 1];
        for (int i = 0; i < solution.length; i++) {
            solution[i] = Integer.MAX_VALUE;
        }
        solution[0] = 0;  // base case
        for (int i = 1; i <=sum; i++) {  // build solution one-by-one
            for (int coin: coins) {
                if (i - coin >= 0) {
                    solution[i] = Math.min(
                            solution[i],
                            solution[i - coin] + 1
                            );
                }
            }
        }
        return solution[sum];
    }

}
{% endhighlight %}

For every next, yet unknown, solution, we find optimal (minimal) number of coins by checking how previous sums are constructed. For example, if a sum to build is 10 and available coins 3 and 5. The best choice for 10 is to take minimum for 7 (10 - 3) or 5 (10 - 5) and add one. Becuase we calculate from left to right, solution[7] and solution[5] are already optimal.

This problem has overlapping subproblems: we use this in top-down solution with memoization.<br/>
This problem has optimal substructure: we use this in bottom-up solution; for every larger problem (large target sum) we iterate through all previous smaller tasks to pick the best.

Bottom line: build brute force recursive solution -&gt; get frustrated with speed -&gt; add memoization -&gt; convert to bottom-up solution by looking into recursion from opposite way.





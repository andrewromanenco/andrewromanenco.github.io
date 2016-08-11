---
layout: post
title:  "Test Driven Development by example"
date:   2016-08-11
tags: [design]
---

TDD is not about having tests with your code. It's not about writing tests before writing code. It's about driving design and splitting coding process into bite-size steps using tests as drivers.

Just a simple example how test-driven-development works.

The goal is to write a function to concatenate two strings. Assume that an input is two non-null strings and a result is also a  non null string.

Let's have our first test for the second part of our statement - non null result:

{% highlight java %}
Func testResultNotNull() {
  result:String = concatenate("a", "b")
  assert_not_null result
}
{% endhighlight %}


Although, the test is trivial, it already contributes to the design. The test has an expectation to the function interface: the input is two strings and the output is a string as well.

This implementation makes the test happy:

{% highlight java %}
Func sum(s1 String, s2 String): String {
  return s1 + s2;
}
{% endhighlight %}

Although, the test is green now, the code above is not TDD. The problem is that the code is way too complicated. Here is a better implementation:
 
{% highlight java %}
Func sum(s1 String, s2 String): String {
  return "";
}
{% endhighlight %}

The test is green as well. But look into the implementation - it does exactly what the test expects and nothing more - the implementation returns non null result. It's is important to mention that first implementation is not just complicated; it has a code path without any test coverage; for example, both "a+b" and "b+a" would keep the test green.

This is, probably(IMHO), the most important aspect of TDD, to make a test green, a developer must take a shortest path.

Now, when we have the test green with minimal possible implementation, it's time for the next step. Make  tests:

{% highlight java %}
Func testTwoStringsAreConcatenated() {
  result:String = concatenate("a", "b")
  assert_equal "ab", result
}

Func testFirstStringIsReturnedIfSecondIsEmpty() {
  result:String = concatenate("a", "")
  assert_equal "a", result
}

Func testSecondStringIsReturnedIfFirstIsEmpty() {
  result:String = concatenate("", "b")
  assert_equal "b", result
}
{% endhighlight %}

The test set from above is a full specification for concatenate() behaviour. It covers all edge cases as well. Why no null input scenario? The interface description states that nulls are not allowed. If someone will send a null, Null Pointer Exception will be raise, and this is ok (defensive programming is a topic for another day).

To make tests from above green, the function's body get modified to:

{% highlight java %}
Func sum(s1 String, s2 String): String {
  return a + b;
}
{% endhighlight %}

Another key point for TDD is the size of each step. Sometimes, a simple test case for a + b may result in a heavy development; imagine that strings must be validated by a remote service. For this cases, current test should be postponed and underlined features must be developed in TDD style. (Postponing is very convenient with Git branching)

Summary:

 - Give descriptive names to tests
 - Write test for a specification or a feature first
 - Cover entire specification and all edge cases
 - Make MINIMAL possible implementation to satisfy tests
 - With everything green, refactor
 - Repeat

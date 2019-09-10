---
layout: post
title:  "Final local variables in java. Some intentions matter."
date:   2019-09-10
tags: [software-development, java]
---
There are three types of developers, when it comes to the use of final local variables in Java: i) Use final local variables when they have to, ii) Use final variables only when they have to; and iii) use final local variables whenever they can.

The first group is represented by developers, who are forced to use final local variables to make the java compiler happy. For example, in some Java versions, an anonymous class may access local variables of an outer code only if those variables are final. Every developer starts as a member of this group.

The third group declares local variables as final whenever possible. Some of these developers configure IDEs to add the final keyword automatically. Both Eclipse and IntelliJ Idea have this setting out-of-box. An example of such code would be:

{% highlight java %}
final String value = “some-value”;
for (final String item: listOfStrings) {
    System.out.print(item);
    System.out.println(value);
}
{% endhighlight %}

The second group is the third group’s opposition. These developers are against declaring every possible local variable as final. Developers join the second group when they become a part of a team with some members being a part of the third group. It takes time to understand the value of the final keyword. Hence, it is natural for a developer to stay in the second group for a while. Sometimes even forever. If they are lucky, they will rise to the third group.

There are pros and cons for using final local variables. They also have three buckets: reasons to avoid the use of the final keyword everywhere, the wrong reasons to use, and the right reasons to use.

The main reason against marking all local variables as final is the verbosity. Java is not a particularly concise language, and adding even more text is a problem. Although this is true, it is a small price to pay for all the advantages outlined below.

The wrong reason to use the final keyword is an attempt to help the compiler with the optimization. Compilers are sophisticated enough to figure out these small optimizations themselves. Actually, in java, there is a notion of effectively final variables. These are non-final variables, but they are never reassigned. The compiler has no trouble to detect such use cases. Thus, this is not a reason to declare every local variable as a final one.

Another wrong reason is insurance against accident variable modifications. Strictly speaking, declaring a local variable as final does solve unintended reuse problem. At the same time, this is the wrong tool. Good unit tests prevent misuse of a variable.

The main argument for using the final keyword whenever possible is communicating the intent. A developer writes code once, and other developers read it many times. Hence, the primary goal is to simplify the reading. Declaring a variable as final sends a strong message - I plan this variable to be a read-only one.

Intensions matter in source code. When a developer creates a method to sort data, the method name is "sort". The reviewer may reason about the method behavior even without looking into the actual implementation. The author's intent is clear. The same principle applies to immutable instance variables. The same principle applies to local variables as well. If a local variable is intended to be immutable, it should be declared that way.

A quote from Java Language Specification (JLS): *4.12.4 A variable can be declared final. A final variable may only be assigned to once. Declaring a variable final can serve as useful documentation that its value will not change and can help avoid programming errors.*

Of course, the final keyword is not the only tool to simplify code reviews. Short methods with good names contribute to simplicity a lot. Unfortunately, short is not a well-defined metric. Most projects have methods longer than ten lines of code. Sometimes it just makes no sense to go smaller. For those cases, using final local variables helps. It makes sense to declare all variables final for consistency. There is no need in a team debate how long a method should be to apply finals.

It takes time to see the value of the final local variables. A reviewer’s brain has to be trained (hello, machine learning) to use the keyword to simplify the mental model. This won’t happen in a day or a week. For instance, studies show that about two months are required for a habit to become automatic. Probably, this is the time when a reviewer’s brain will rewire itself.

Additional arguments for using finals come from other languages. Kotlin and Scala are two examples from many. Both have variables and values. Variables are declared with "var" keyword, and values are declared with "val" one. "val" declares a variable, which can be assigned exactly once. Kotlin calls these variables - read-only. Hence, this is a final variable in java’s world. The reason why final variables are the first-class citizens in other languages is the same - this helps to build a mental picture of the code under review.

Java 10 brings a new way to declare variables - the "var" keyword. Unfortunately, there are no plans for a "val" one. Seems like "final var" will be out there for a while, for those who get it.

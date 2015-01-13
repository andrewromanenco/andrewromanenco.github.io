---
layout: post
title:  "Return statement in try-finally block in java"
date:   2015-01-06
tags: [java]
---

What will this program print out?

{% highlight java %}
public class Test {

    public static void main(String[] args) {
        System.out.println(method());
    }

    public static int method() {
        try {
            return 1;
        } finally {
            return 2;
        }
    }

}
{% endhighlight %}

Result is 2.

We know that *finally* gets ALWAYS executed (unless jvm died). So, out method returns 1 and then, finally is called. So it returns 2 and overrides previous value. Question is why?

The answer is in the approach used by jvm for method calling.

When a method is called, jvm creates a stack frame. When a method is done, it's stack frame is removed and the result is pushed to the caller's stack.

Returning an integer from a message call is represented by ireturn byte code instruction (hex code #AC). Obviously, it's not possible to push more than one result to a caller's stack - it would break indexing and raise a runtime error. To protect against this failure, the compiler actually removes first return statement and keeps byte code instruction for second one only.

Here is javap result for the example from above:
{% highlight java %}
public class com.Test {
  public com.Test();
    Code:
       0: aload_0
       1: invokespecial #8                  // Method java/lang/Object."<init>":()V
       4: return

  public static void main(java.lang.String[]);
    Code:
       0: getstatic     #16                 // Field java/lang/System.out:Ljava/io/PrintStream;
       3: invokestatic  #22                 // Method method:()I
       6: invokevirtual #26                 // Method java/io/PrintStream.println:(I)V
       9: return

  public static int method();
    Code:
       0: goto          4
       3: pop
       4: iconst_2
       5: ireturn
    Exception table:
       from    to  target type
           0     3     3   any
}

{% endhighlight %}

Basically, if you have a return statement in a finally block - it will be the only one return point in the compiled code.

---
layout: post
title:  "Reload a java class at runtime"
date:   2014-12-20
tags: [java, jvm]
---

 * [What a classloader is](#1)
 * [Theory](#2)
 * [Developer's point of view](#3)
 * [Implementing a dynamic classloader](#4)

<br/>

### <a id="1"></a> 1.

Before a Java Virtual Machine can use a class, someone has to provide a representation of that class. Classloaders are those intermediaries to decouple jvm from knowing how a class was created and loaded. On a high level, a classloader has to provide an instance of java.lang.Class, based on requested fully qualified name (package + class name).

Common examples are: loading a class from a file or jar, load a class from a web server, generating a class on the fly based on some conditions and, of course, dynamic class loading to switch a class at runtime.

<br/>

### <a id="2"></a> 2.

Although responsibilities for classloaders are simple: provide a java.lang.Class from a String name; there are several conventions how a resolution process should proceed.

Here is how this process looks like: jvm has a hierarchy of classloaders; and each, but the top one, has a parent (parent has no knowledge about its children). When a classloader is asked to discover a class, it asks his parent first. Only if parent fails, it tries to load a class itself (and fail with ClassNotFound exception if not found). Easy to see that this ask-first approach forwards each request to top, until no more parents are available.

A question one may ask: if a classloader receives a string as a request for a class loading, who is that class loader who gets a request to load "java.lang.String" itself? Even more, a class loader extends java.lang.ClassLoader - so someone has to load it as well. To load a class loader you need to load a class loader, load a class loader, load a class loader...(recursion detected).

There are three levels for these classloaders: bootstrap, extensions, system (aka application) and user. Bootstrap loader is a not java class (break the recursion), it gets created by a jvm as a native implementation and is responsible for loading classes from rt.jar and others (check lib in you jre folder). Extensions loader checks ext folder and system one is loading classes from classpath. User's classloaders reside under the system one and execute custom goals you need.

Having this process in mind, it's easy to explain how java.lang.String is loaded into jvm's memory: your code asks for an instance of String, system classloader receives a request for the class name and sends it up to its parent; and so on until bootstrap loader being asked. Bootstrap will return instance of java.lang.Class (from rt.jar), representing java.lang.String.
Now let's try to load our own class (e.g. test.OurApp). As usual, system class loader delegates the request up, to extensions and, upper, to bootstrap. Of course, bootstrap loader has no idea about our application and will throw ClassNotFoundException, next layer catches the exception and also try-fail to discover the class. At the end, system class loader catches the exception and tries to locate the class in the classpath. Assuming it's there, the class gets loaded and our code can keep running.

<br/>

### <a id="3"></a> 3.

Now let's take a look into this process as developers! Find source for java.lang.ClassLoader and scroll to loadClass method. This method is called when a class is required and has several steps:

 1. check if a class is already loaded(uses native call, findLoadedClass0), and, if yes, return the class
 2. ask parent to load the class (and catch ClassNotFoundException)
 3. try to load the class itself by calling findClass (hello template method pattern,   see more details in below, in implementation)

One more very important detail: what does it mean that a class is already loaded? To answer this question it makes sense to notice that an identification of a class at runtime is a combination of its class name and a classloader. So a class can be loaded by more than one classloader and (obviously) a class loader can load more than one class. To make sure all these combinations work at runtime, every time you access a class the jvm check if classes are compatible by checking both the class name and the classloader.

Theory says: a classloader L who created a class C is it's defining loader. If L creates C directly or by delegating to its parent L is initiating loader for C. And jvm tracks combinations of class names and defining loaders to make "same class". And, of course, this makes it possible to load the same class twice, but with different defining classloader.

 It's interesting that developers definition for "same class" is different from jvm's one. For a developer same classes are (usually) those who have same name and same interface, but different behaviour or implementation.

<br/>

### <a id="4"></a> 4.

Knowing theoretical part, we can build small application to reload a class at runtime. The source code is available <a href="https://github.com/andrewromanenco/classloader-example" target="_blank">@github</a>, let's review it step by step.

First of all we should design how our code is be structured. Three source folder are there:

 * src - our main classes, they are available in classpath at runtime and are loaded by system classloader
 * src_dynamic_1(2) - these are our implementations for our reloadable class dynamic.DynamicClass. Dynamic classes are not part of our classpath at runtime (so they are not available to system classloader)

Our dynamic classes are simple. Basically, all that they can do is to print a string to identify which one was called. Make sure you noticed that they implement same interface and this interface belongs to application code, is part of runtime classpath and is loaded by system classloader. This is important. You cannot have DynamicClass variable in your source code as it will be loaded by system classloader. Actually, it will even fail at resolution as dynamic classes are not visible (are not in classpath).

To explain it even deeper. Let's assume that a class X is in your classpath and you have line

{% highlight java %}
X x = loadSomeX();
{% endhighlight %}

As soon as this line gets executed, jvm will load X from the classpath. Now if your custom classloader will try to load other version of X, you will get an error - class cast exception. Why? Because X with system classloader is not the same as X with other classloader. So our choices are either cast everything to Object or use reflection api. Both ways are ugly.

To resolve issue from above, we can and should use interfaces. An interface is identified by its name and as long as dynamic code implements it, it is safe to assign loaded class instance to interface variable.

Check these <a href="https://github.com/andrewromanenco/classloader-example/blob/master/src_dynamic_1/dynamic/DynamicClass.java" target="_blank">classes</a> (<a href="https://github.com/andrewromanenco/classloader-example/blob/master/src_dynamic_2/dynamic/DynamicClass.java" target="_blank">2</a>) and the <a href="https://github.com/andrewromanenco/classloader-example/blob/master/src/com/romanenco/cloader/Action.java" target="_blank">interface</a>.

After we are done with our reloadable code, we can implement the loading process: remember - to load same class more than once we have to use different classloaders. To achieve this goal we will make our classloaders configurable with custom classpaths. And, at runtime, we will create two classloaders, so that different versions of DynamicClass are available.

Take a look to our class loader <a href="https://github.com/andrewromanenco/classloader-example/blob/master/src/com/romanenco/cloader/CustomClassLoader.java" target="_blank">implementation</a>. We set parent class loader and class path in the constructor. Also we replace find class with our own implementation to actually read .class file from custom classpath. FindClass(…) is the only method to override; this is due to default implementation of loadClass(…) - ask parent first, if no class found, try itself.

Actual byte code loading is <a href="https://github.com/andrewromanenco/classloader-example/blob/master/src/com/romanenco/cloader/file/ByteFileReader.java" target="blank">trivial</a> - we have to provide byte code as an array.

Now let's see how it works at runtime. Check out <a href="https://github.com/andrewromanenco/classloader-example/blob/master/src/com/romanenco/cloader/Example.java" target="_blank">an example</a>. It's pretty strait forward. It loads specific implementation of DynamicClass from specific class path by using right classloader instance.

Make sure you checked ant build script for the project. It has separate goal for compilation and execution; and these targets have different classpaths.

### Final notes

Many java containers work very similar to the code from above. For example tomcat. When you deploy your app into webapp folder, your code is outside of tomcat classpath. Tomcat creates new classloader and points it to your webapp. And the entire process works as described above. When you redeploy your app, tomcat just deletes current classloader and creates new one, so new versions of your classes are read.

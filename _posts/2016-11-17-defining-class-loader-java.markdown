---
layout: post
title:  "Defining class loader in Java"
date:   2016-11-16
tags: [java]
---
Java uses classloaders to load classes on demand based on their names. The context is simple: there are three class loaders: bootstrap, extensions and system; and there is a delegation model in place: when a classloader is asked to load a class, it should first ask parent and, if no class is found, try to load the class itself.

**It seams, there is a confusion how java makes a choice which classloader to use when a new class is required to be loaded.**

If a class C has a reference to another class D (for example, D is super of C), then the defining loader of class C is going to be asked to do the job.

Definition for Defining Classloader is simple: a class loader instance, which run .defineClass(name,…) method is recorded as class's defining classloader and is available via .getClassLoader().

To illustrate this idea, we can write this code: <a href="https://gist.github.com/andrewromanenco/290b6b4ad8111496eebf36f8e1e991f3" target="_blank">view gist</a>.

First of all, few words about .defineClass() method. It accepts bytes and generates a class as a result. These bytes are usual java bytecode of a compiled class. Let's write a class:

{% highlight java %}
class Custom {
    public static final int I = 99;
}
{% endhighlight %}

Now we need two classloaders to demonstrate delegation model. First class loader is a straightforward one, it just prints calls to loadClass and findClass methods:

{% highlight java %}
class DelegatingCL extends ClassLoader {

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
        out.println("CLoader.loadClass(s, b): " + name + ", " + resolve);
        return super.loadClass(name, resolve);
    }

    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        out.println("CLoader.findClass(s): " + name);
        return super.findClass(name);
    }

}
{% endhighlight %}

Second classloader uses delegation for all class names, other than our custom class. If a requested class name is "Custom", the code is going to read the file and call defineClass() - which makes it defining classloader for Custom.

{% highlight java %}
class InterceptingCL extends ClassLoader {

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
        out.println("CLoader.loadClass(s, b): " + name + ", " + resolve);
        if (name.equals("Custom")) {
            return readAndDefine(name);
        }
        return super.loadClass(name, resolve);
    }

    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        out.println("CLoader.findClass(s): " + name);
        return super.findClass(name);
    }

    private Class<?> readAndDefine(String name) {
        Path path = Paths.get(name + ".class");
        byte[] data;
        try {
            data = Files.readAllBytes(path);
        } catch (IOException e) {
            throw new RuntimeException("Can't read file", e);
        }
        return defineClass(name, data, 0, data.length);
    }
}
{% endhighlight %}

Let wire everything together:

{% highlight java %}
public class CLoaderExample {

    public static void main(String[] args) throws Exception {
        out.println("With delegation");
        final DelegatingCL dcl = new DelegatingCL();
        Class<?> c = dcl.loadClass("Custom");
        out.println("Class loader of C: " + c.getClassLoader());
        Class<?> cParent = c.getSuperclass();
        out.println("Class loader of super: " + cParent.getClassLoader());

        out.println("With interception");
        final InterceptingCL icl = new InterceptingCL();
        Class<?> d = icl.loadClass("Custom");
        out.println("Class loader of C: " + d.getClassLoader());
        Class<?> dParent = d.getSuperclass();
        out.println("Class loader of super: " + dParent.getClassLoader());
    } 
}
{% endhighlight %}

The example uses two different classloaders to load the same class.

This is important to note that loading of Custom class requires it's parent to be loaded as well, which is java.lang.Object.

With delegating class loader, the code prints log message for loadClass call. But the call is delegated to the parent (which is system class loader in this case) and the parent creates the class via calling defineClass. This means that system classloader is now the defining class loader for Custom.

During Custom loading, Object has to be loaded as well. Java picks Custom's defining class loader to do the job. So, system class loader is used to load Object, and this call is delegated up. Object's class loader is null, which means bootstrap one.

With intercepting class loader the flow is different. loadClass is called and it loads bytes directly, followed by call to defineClass(). This makes InterceptingCL the defining class loader for Custom. Custom needs it's parent to be loaded and defining class loader will be asked to get it. That's why there is another call to loadClass. Of course, the actual load is delegated and defining loader for Object is bootstrap one.

Bottom line: to figure out which classloader is going to be called to load a class, always think who did call defineClass.

How to run this locally:

    - dump all code into single file CLoaderExample.java
    - Use command line to compile it: javac CLoaderExample.java
    - Use command line to run: java -cp . CLoaderExample



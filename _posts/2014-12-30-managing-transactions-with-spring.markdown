---
layout: post
title:  "Managing transactions with Spring"
date:   2014-12-30
tags: [java, spring]
---

Here is how one can configure transaction management in a java application with
Spring framework.

### Introduction

Four cases will be explained and demonstrated:

 * no transaction management
 * local transaction with declarative configuration
 * local transaction with programmatic approach
 * distributed transaction with declarative configuration

Prerequisites:

 * java 7
 * maven 3
 * any J2EE server (for example WildFly/JBoss)

Please, refer to source code at <a href="https://github.com/andrewromanenco/transactions-example" target="_blank">
github</a>

### Application

This application is a maven based web project.

For simple, local transactions single datasource is in use (this is a HSQLDB in
memory instance). JUnit is more than enough to demonstrate how the code works.

It’s a bit more complicated with distributed transactions. First of all, another
datasource is required - this is also a HSQLDB database. The code is called from
a web controller - see distributed transaction section below.

Common code for all samples is represented by
<a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/main/java/com/romanenco/transactions/Service.java" target="_blank">Service</a>
and
<a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/main/java/com/romanenco/transactions/DAO.java" target="_blank">DAO</a>
(and it’s implementations). Service is the major piece of our app, it updates
both datasources, so someone (spring) has to manage all these updates in a
single scope.

### No transaction management

This is very basic use case: no specific configuration is submitted and datasource operates in auto-commit mode; all inserts are independent. Check <a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/test/java/com/romanenco/transactions/NoTXTest.java" target="_blank">sample unit test</a>.
Focus on testFail(): ltx.fail() will try to insert three records, with second
record failing due to table constraints. But two other rows are still inserted.
This is auto-commit.

### Local transaction with declarative configuration

Let’s assume that our ltx.fail() should actually be in a transaction - if one of
inserts fails, we want all of them to be rolled back as well. Here spring comes
to play. Instead of making any changes to our code, we should just give spring
specific instructions: when a transactions should be started and when it should
be committed or rolled back.

Refer to spring <a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/test/resources/local-context.xml" target="_blank">configuration</a>.
To tell spring what to do, the config is using aop - aspect oriented
programming. Basically, we tell to apply a rule (tx-advise), when a specific
condition is met (app:config). In this case, our condition is a call to LocalTX
instance (any method); and the rule is rollback when a Throwable is detected. I
assume you know that any exception is a child of Throwable.

Check out this <a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/test/java/com/romanenco/transactions/LocalTXTest.java" target="_blank">unit test</a>.
In this case, LocalTX instance is received from Spring and, actually, is a
proxy. So when we are calling ltx.fail(), Spring knows that transaction has to
be started. Our second insert in fail() is throwing DuplicateKeyException which
is instance of Throwable; this triggers full rollback.

###Local transaction with programmatic approach

If, for some reason, AOP is not an option for your project, you can achieve the
same result using hand-coded solution. Refer to <a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/test/java/com/romanenco/transactions/LocalTXProgramTest.java" target="_blank">LocalTXProgram</a> - it’s pretty straight forward.

### Distributed transaction management

Distributed transaction is a transaction which spans across multiple resources,
e.g. two databases. Usual TRANSACTION BEGIN, COMMIT/ROLLBACK way is not working
any more. The reason is in fault tolerance. It is always possible to find a
sequence of events so that one datasource gets committed and other one gets
rolled back.

Distributed transaction manager is required to make sure that all resources are
either committed or rolled back. It is possible to use standalone manager, but
for our example it’s way easier to use embedded one. All J2EE containers have
one.

Refer to our spring <a href="https://github.com/andrewromanenco/transactions-example/blob/master/src/main/webapp/WEB-INF/springapp-servlet.xml" target="_blank">mvc config</a>.
Major difference to all previous configs is declaration of
JTATransactionManager. This manager gets connected by spring to an embedded one
via jndi.

Take a look to this <a href="https://github.com/spring-projects/spring-framework/blob/master/spring-tx/src/main/java/org/springframework/transaction/jta/JtaTransactionManager.java" target="_blank">spring-tx source</a>.
When spring tries to discover existing service in JNDI, this list of names is used:

{% highlight java %}
public static final String[] FALLBACK_TRANSACTION_MANAGER_NAMES =
            new String[] {"java:comp/TransactionManager", "java:appserver/TransactionManager",
                    "java:pm/TransactionManager", "java:/TransactionManager"};
{% endhighlight %}

If no manager is found (for example, if WAR is deployed to a tomcat), you will
get an error message:

{% highlight java %}
IllegalStateException: No JTA UserTransaction available
{% endhighlight %}

To make sure JTA is available, the WAR should be deployed into J2EE server. For
example, wildfly/jboss. Download full version, run it as standalone instance
(bin/standalone.sh) and copy WAR to standalone/deployments. After the app is up
and running, open test page: localhost:8080/transactions-example/test.do . Of
course, the app can be deployed to any other J2EE server (e.g. weblogic)

In general, it works exactly the same as local transaction configuration. The
only difference is use of JTA manager.

Worth mentioning, that the most common use case for distributed transaction is
JMS + DATABASE. If a message from a jms source is not saved into a database, it
should be put back.

Make sure you read about <a href="http://en.wikipedia.org/wiki/Two-phase_commit_protocol" target="_blank">two-phase commit protocol</a>.


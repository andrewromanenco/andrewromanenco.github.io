---
layout: post
title:  "Transaction manager in Spring Framework"
date:   2015-02-15
tags: [java, spring]
---

<a href="{{ post.url }}/managing-transactions-with-spring" target="_blank">As you know</a>, it's is not too difficult to configure transaction management using Spring framework. The only thing, worth to mention, is implementation details, how Spring handles this process.

***Spring HAS NO BUILT-IN TRANSACTION MANAGER***

For some reason, most developers think that Spring can handle everything; but, actually, this is just an integration layer. When an application activates transaction layer; Spring scans JNDI for a transaction manager, using list of predefined names. These names cover most JEE servers (e.g. weblogic).

Transaction manager discovery class is org.springframework.transaction.jta.JtaTransactionManager and the list of names is

{% highlight java %}
public static final String[] FALLBACK_TRANSACTION_MANAGER_NAMES =
            new String[] {
            "java:comp/TransactionManager",
            "java:appserver/TransactionManager",
            "java:pm/TransactionManager",
            "java:/TransactionManager"};
{% endhighlight %}

{% highlight java %}
// Check fallback JNDI locations.
        for (String jndiName : FALLBACK_TRANSACTION_MANAGER_NAMES) {
            try {
                TransactionManager tm = getJndiTemplate().lookup(jndiName, TransactionManager.class);
                if (logger.isDebugEnabled()) {
                    logger.debug("JTA TransactionManager found at fallback JNDI location [" + jndiName + "]");
                }
                return tm;
            }
            catch (NamingException ex) {
                if (logger.isDebugEnabled()) {
                    logger.debug("No JTA TransactionManager found at fallback JNDI location [" + jndiName + "]", ex);
                }
            }
        }
{% endhighlight %}

P.S.

Tomcat is not a full JEE server and has no transaction manager.
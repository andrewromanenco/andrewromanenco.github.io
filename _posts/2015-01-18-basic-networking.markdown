---
layout: post
title:  "Basic networking (ARP, DNS, TCP/IP)"
date:   2015-01-18
tags: [stuff]
---

`This is post is about high level networking.`

Here are layers for TCP/IP:

 * Application layer: adds logical meaning to data
 * Transport layer: allows to have more than one communication line with the same ip address
 * Internet layer: identifies target for data deliver
 * Link layer: responsible for connection in local network


### ARP

After a name has been resolved to an ip address (with DNS, see below), we, probably, want to start exchanging network packages with the remote machine. Let's assume that we keep communication with port 80 (web server).

Our desktop is connected to the internet. What does it mean? Desktop is actually connected to local network, this network is connected to your provider's link and your provider is connected to other global networks. There is a well-defined mechanism how all these connections work together to deliver data between any two points.


## Send data to a computer in the same network

On hardware level all computers are identified with MAC addresses. So when one wants to send some data to a specific box identified by IP address, IP-to-MAC resolution must be executed. Address Resolution Protocol is design to achieve this goal.

ARP executes these steps to resolve an IP address to specific MAC; first of all the cache is checked. If the IP-to-MAP pair is there, it is used for all communications. If the cache is empty, ARP sends a specific broadcast message with IP address inside. This message is sent to MAC: FF:FF:FF:FF:FF:FF. Every computer in local network receives this message and can see if its own IP address matches the one requested. If yes, the response is sent out and the resolution is done

Another use case for ARP is checking for duplicated IP addresses in given local network. When a new computer is connected to a network, it sends out a probe request with his desired ip address. This request is also a broadcast. So all other computers receive the message and are able to reply if there is IP address conflict.

## Send data to a computer in other network

When a data has to be sent to a computer in remote network, all packages must travel through a router (or routers). ARP resolution is still required. Client makes a decision based on IP address if a destination is part of local network. If it is, the process from above is used. But if a destination is in remote network, it is known that all data must be sent to a router.

Decision if a destination is local is made based on IP address and MASK. Mask is just a binary number with some of lower bits set to zero. If our own IP address is IP1 and the remote is IP2; then the logic works this way:

A = IP1 and MASK<br/>
B = IP2 and MASK<br/>
if A == B then both computers are in local network<br/>

After it is confirmed that the target is in a remote network, router resolution must take place. Each computer is configured with default router IP address. And it is used for all communications, unless more specific routes are there.


## IP vs TCP/UDP

IP address is property of Internet layer of TCP/IP stack. It allows effective routing and data delivery between two parties. The only thing which is missing is ability to support more than one connection between two parties at the same time. To resolve this issue, we have to move layer up and check for TCP or UDP protocols. These protocols add concept of port on top of internet layer. Now, with ports defined, many connections are manageable at the same time.


## DNS

Name to ip resolution is always first step in making a connection to a named server. Domain Name System (DNS) is designed to achieve this goal in very effective way.

First thing to know about DNS is that the system is distributed. With more than 250 millions of domains registered (as of 2012), there is no way any single server is able to support that. Management of this number of items is also not a trivial problem.

DNS addresses these issues by creating a hierarchy: small set of root servers delegates branches to different sub-servers; and this process is repeated in recursive approach. As a result, there is a three of all domain names registered in the internet.

After the tree exists, it's easy to see how our resolution requests are processed. To get an ip address for a name (for example a.b.c.com) dns client sends requests to highest known server (in this case for com domain). Each server will respond with IP address we are interested in or with an address of other sub-server who might know the answer. So the dns client goes to that sub-server and repeats the same steps. This process works until domain is resolved or no more sub-servers exist.

Of course, in real life, many DNS servers cache answers for specific period of time to server clients better. But the main idea is to ask DNS tree from root to its leafs until the answer is found.


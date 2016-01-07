---
layout: post
title:  "Bitcoin payment walkthrough"
date:   2016-01-06
tags: [cryptography]
---

Lets assume Alice has 11 bitcoins and wants to transfer 10 of them to Bob. Here is how the payment happens:

 - Alice creates a transaction
 - Alice broadcasts the transaction to bitcoin network
 - A random node picks Alice’s transaction (and other unprocessed ones) and creates a block
 - Block gets added to bitcoin’s blocks chain

### Alice creates a transaction
Having a bitcoin means that there was one or more transactions where a user(his/her public key) was set as a receiver. For example, there are two different persons sent 5 and 6 bitcoins to Alice. Now, with total of 11 bitcoins, Alice sends 10 of them to Bob.

She creates a transaction:

 - specifies both input transactions (5 + 6 = 11 coins total)
 - specifies Bob’s public key as a receiver of 10 coins
 - specifies Alice’s own public key as a receiver of 1 coin
 - signs the transaction data with Alice’s private key

It worth to note that input and output amounts always match.

### Alice broadcasts the transaction to bitcoin network
Bitcoin network is a set of all computers (nodes) running bitcoin software. Each node keeps connections to several neighbors; neighbors have their own neighbors and so on; it’s a peer-to-peer network. Each node may have it’s own view of the universe, but eventually they all get to the same page and have access to same transactions. After Alice creates a transaction; this transaction is distributed over the network using very simple algorithm: Alice’s node tells every neighbor about this transaction; and neighbors forward the data down the network. At some point, all (or most) nodes will have Alice’s transaction in their lists of unprocessed records.

Each node is an independent one, which means that each node makes its own decision, if proposed transaction is valid. There are two common cases when a transaction is invalid.

 - First of all, the digital signature may be wrong; usually this means that someone attempts to steal coins. These transactions get rejected.
- Second case is double spending attack. Someone may try to spend same coin more than once. There is a check against that. Every node picks inputs for a proposed transaction; and inspects blocks chain to make sure those inputs has not be spent yet. Note, there is no need to go back to the beginning of the history. Each input has a pointer to owning block and the node checks all blocks appeared after.

### A random node creates a block
A block in bitcoin has this data:

 - a pointer to previous block (other than very first, genesis block)
 - a list of transactions (moving bitcoins between accounts)
 - special transaction to create new bitcoin and assign it to an account

Every node would like to create blocks, due to reward of bitcoin creation. To make sure this process stays predictive and rate of coins creation is stable a proof-of-work algorithm is used.

Proof of work is a target which is difficult to calculate, but once it’s calculated, it’s trivial to check its correctness. Every node picks unprocessed instructions and makes a block. This block is identified by hash - H. And there is a number NONCE. Proof of work is this function:

`t = HASH(H, NONCE)`

Basically, this calculates another hash, based on a hash of a new block and a number. If `t` is less than predefined `target`, then the block is considered as a valid one and is sent to entire network. If `t` is greater, than the `target`, NONCE value get incremented, until `t` is in the range. Upon receiving a new block each node checks proof of work, by re-calculating `t` and comparing it to `target` value. This is an insurance that invalid blocks will be rejected.

### Block get added to bitcoin’s block chain
Bitcoin network is configured to generate a new block once every 10 minutes. If the software identifies that blocks are generated more often, it decreases the value of `target`, making the calculation more complicated. The complexity of calculating proof of work guaranties random node selection according to calculating power available.

It is possible that more than one node will have proof of work at the same time. In this case both blocks will be valid; both will be distributed until one of them reaches more than 50% of available nodes; the other one get rejected.

<br/><br/>
[Bitcoin wiki is a great resource to dive into this way deeper.](https://en.bitcoin.it/wiki/Main_Page)


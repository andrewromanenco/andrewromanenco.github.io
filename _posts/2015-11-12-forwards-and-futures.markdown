---
layout: post
title:  "Forwards and Futures"
date:   2015-11-12
tags: [finance]
---

Forwards and futures are derivatives: financial instruments with prices depending on underlying assets.

Explanation of forwards and futures should be given in a context of a cash flow. For example, let assume there is a process (cash flow) consuming fuel and producing some kind of product. To be more specific, the process:

 - lasts for 4 months
 - in the beginning of each month it consumes 25 litres of fuel
 - at the end of four months it will produce 20 items of a product
 - one litre of fuel costs 1$
 - current price of one item of the product is 10$

If everything goes according to the plan, at the end of 4 months 25 * 1 * 4 = 100$ will be spent on the fuel. When the product is going to be sold on the market, it will bring 200$. Which means that the outcome of this cash flow is +100$ (difference between money out and money in).

Although this process does look attractive (positive cash flow), there are several risks involved. Prices are determined on the market and  fluctuate with time. To keep things simple, lets assume the price for the fuel is a constant (otherwise we would have two forwards/futures instead of one). With this assumption in place, there are two possible outcomes:

 - product price stays the same or goes up: the profit stays as planned or grows up
 - product price goes down: the profit goes down as well, and a loss is possible

Choices are either to accept the risk of lower prices (basically, gamble) or try to handle it.

Forward is a solution to mitigate price changing risk. Basically, one should reach clients and get to an agreement of selling 10 items of the product for 20$ in four months from now. It worth to mention, that this contract handles price changing risks for both parties; both sides are protected from price movements. When time comes, forward contract get executed by shipping agreed amount of the product and receiving agreed payment.

Forward contract introduces other risk. What if the the client goes bankrupt and will not be able to pay for the product? There is no much one can do. Because forward is a contract between two sides, it’s usually highly customizable by both parties involved; and it usually has no value for anybody else. Counterparty risk is always a part of any forward contract.

Future is a solution to handle counterparty risk along with price risk. As usual, handling a risk does not come for free.

First of all, futures are standardized contracts. It has to be a market for the product you want to have future with. All details of that contract are know upfront and you can’t customize it for your needs. These rules are regulated and future exchanges exist to make trading of future contracts possible. Of course, having an exchange as an intermediary brings up additional costs.

Secondly, futures have completely different mechanics comparing to forwards. Futures mechanics can be illustrative with an example. One wants to sell 10 items of a product for 20$ in four months from now. This contract can be acquired on an future exchange for a fee. Note, there is no counterparty involved, it’s just an agreement with futures exchange (FE from now). Basically, FE guaranties that on execution of the contract, you will be able to receive 200$, whether the price going up or down. To make this possible FE runs mark-to-market process on daily basis. Marking process is simple, it take current price for the product and compares it with price agreed in the contract. There are two options here:

 - price goes down, for example to 10$: if the seller would sell the product right now, the received payment would be 100$; which is 100$ less than expected amount of 200$. To handle this, FE will put missed amount of 100$ to seller’s account.
 - price goes up, for example to 30$; if the seller would sell the product right now, the received payment would be 300$; which is 100$ more than agreed amount of 200$. To handle this, FE issues margin call and asks seller to put the difference to his margin account. The seller have to put 100$.

This process repeats on daily basis with money coming to and from sellers margin account. And on contract execution date, the seller will ship the product and receive amount of money agreed upfront.

To sum it up.

Forward is a contract between two parties to have a deal in the future for a price set upfront. Because it’s between two parties only, the contract is customizable, but it carries counterparty risk (risk of other party to fail contract execution). Execution of a forward  contract is as simple as product delivery and payment processing.

Future is a standard contract acquired on an exchange. It is secured through a process of comparing agreed price to market one on daily basis; with money coming to and from margin account. Execution of a future contract involves selling product on the market for a price available at that date; with option to be compensated by the exchange is case of lower prices and paying extra money to the exchange if prices are higher.

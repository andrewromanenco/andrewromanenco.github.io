---
layout: post
title:  "instanceof in JavaScript"
date:   2015-12-07
tags: [javascript]
---

JavaScript is a prototype based language. It supports parent-child object relations. If the relation is implemented in the right way, operator ```instanceof``` works as expected:

{% highlight javascript %}
var Parent = function(param) {
  this.p_value = param;
}
Parent.prototype.p_func = function() {
  console.log('call parent_func');
}

var Child1 = function(param, param2) {
  this.c_value = param2;
  Parent.call(this, param);
}
Child1.prototype = Object.create(Parent.prototype);
Child1.prototype.constructor = Child1;
Child1.prototype.c_func = function() {
  console.log('call c1_func');
}

var Child2 = function(param, param2) {
  this.c_value = param2;
  Parent.call(this, param);
}
Child2.prototype = Object.create(Parent.prototype);
Child2.prototype.constructor = Child2;
Child2.prototype.c_func = function() {
  console.log('call c2_func');
}


var p = new Parent('pp');
var c1 = new Child1('p1','c1');
var c2 = new Child2('p2','c2');


console.log('c1 instanceof Parent is ' + (c1 instanceof Parent))
console.log('c1 instanceof Child1 is ' + (c1 instanceof Child1))
console.log('c1 instanceof Child2 is ' + (c1 instanceof Child2))
console.log('c2 instanceof Parent is ' + (c2 instanceof Parent))
console.log('c2 instanceof Child1 is ' + (c2 instanceof Child1))
console.log('c2 instanceof Child2 is ' + (c2 instanceof Child2))
console.log('p instanceof Parent is ' + (p instanceof Parent))
console.log('p instanceof Child1 is ' + (p instanceof Child1))
console.log('p instanceof Child2 is ' + (p instanceof Child2))

console.log(p.p_value);
console.log(c1.p_value);
console.log(c2.p_value);
console.log(c1.c_value);
console.log(c2.c_value);
{% endhighlight %}

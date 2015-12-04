---
layout: post
title:  "Building a compiler with a virtual machine (in JS)"
date:   2015-12-04
tags: [javascript, compilers]
---

Every developer must implement a compiler at least once in his/her professional life!

Standard way to build a compiler/interpreter contains most of these steps:

 - create/pick language specification
 - create/pick machine instructions specs (e.g. java byte code)
 - create compiler according to lang specs
 - create a code generator for the machine code
 - implement a virtual machine to execute the machine code

In this article, these steps are going to be covered:

 - pick brain*ck as a language specification
 - create a virtual machine specification
 - implement a virtual machine
 - implement a compiler: brainf*ck to machine code
 - do it in javascript: this is due to the book “JavaScript: The Good Parts”, which I read recently. Now ```((jquery === javascript) === true)``` is a false statement for me.

## Language specification

Language specification describes how a language looks like: commands, operations, syntax and so on. A developer must learn language specification to be able to code.
Brainf*ck is a good example for lang spec. It’s short, just eights commands. But implementing it right requires same set of implementations as any other language (e.g. c++).

[Please, refer to wikipedia page for the Brainf*ck specs.](https://en.wikipedia.org/wiki/Brainfuck)

There is one feature to pay attention to: brackets. These are conditional jumps in brain*ck; they always come in pairs, and it’s a mistake to submit a program with unbalanced ones.


## Virtual Machine specification

For now we are going to roll forward to Virtual Machine (VM) specification.

Specification is even more simple than Brainf*ck itself, only these operations are supported:

- MOVE (<>), this operation moves a memory pointer to += steps (steps can be a negative number); basically this moves the pointer to the left or to the right (or does not move it at all)
- UPDATE(+-), this operation update current byte (the one at the memory pointer) to += value; the value can be any number (think how would you handle overflow)
- PRINT, READ - either sends current byte to an output or reads one from an input
- IFJUMP - conditional goto; if current byte is zero, the memory pointer is set to a specified value
- JUMP  - unconditional goto; just set the memory pointer to a specific value

These operations is going to be submitted to a VM as a list (or array). Each operation is going to have an index is the list; with first operation indexed as 0. VM is going to execute operations one by one, until it meets IFJUMP or JUMP.  IFJUMP may change the control flow, JUMP aways changes the control flow; so instead of moving to the next operation in the list, the next operation to execute will be identified by index value (this value comes with the operation itself).

Also, the VM is going to have user's data. This is represented by a continuous array. On every moment, one of elements is identified by memory pointer. The value of this pointer gets updated as operations get executed.

VM specs are very different from language specs:

- not all operations from one spec exist in other one (vm has goto, lang has none)
- vm spec has no operators to go in pairs; IFJUMP and JUMP always know where they have to move the control to

With VM specs set, it’s possible (in theory) to compile a program in any other language; and run it on any VM supporting these specs.

## Virtual Machine implementation

[Here is JavaScript implementation for the specs from above.](https://github.com/andrewromanenco/brainfk-js/blob/master/js/vm.js)

- Memory object is user's data. It has pointer to current array element and methods to move the pointer or update the value. And it is implemented as a hash to optimize memory usage
- TicksLimit: puts a limit how many operations a VM may execute, until exit. This is implemented due to nature of many Brainf*ck program - they never terminate
- VM is a virtual machine implementation. It accepts byte code as a list (according to vm specs); it has program counter (pc) to point to an operation to be executed next. Main logic is in while loop; it just executes current operation and move to next one (or jump)

Note: because vm specs does not support function calls, the VM has no frames stack (google what is frame stack FYI).

## Just compile it

There is a gap between language specification and vm specification. A compiler has to be built to convert human readable Brainf*ck code (haha) to vm instructions. There are several steps to make: tokenize the source, build abstract syntax tree, generate VM instructions.

### Tokenize it

Every language has words. Tokenizer is a tool to take an input source code as a single string and split it into a list of words. For example, we could tokenize this paragraph, and result would be a list [‘Every’, ‘language’, 'has’]. When building a list, there are three groups of tokens: valid ones - we want them to be in the result list, whitespaces - we can ignore them, all others - these are errors and they should be reported.

There are only two groups of tokens in Brainf*ck: valid ones (those eight symbols from lang specs) and white spaces - all others to be ignored.

[Tokenizer implementation.](https://github.com/andrewromanenco/brainfk-js/blob/master/js/tokenizer.js)

The implementation is very simple, two regular expressions: one for valid symbols and one for white spaces. Although in this particular use case, white spaces are all those symbols which are not in valid set; it’s still nice to have white spaces as an explicit definition.

### Parse it
Previous step was about splitting a source to separate word/tokens. Parsing is about figuring out what do words mean.

The result of a parsing is an AST: abstract syntax tree. This is a real tree where nodes are objects, representing operations; and object's properties are operands. [See wiki for details.](https://en.wikipedia.org/wiki/Abstract_syntax_tree)

There are many different technics to produce a valid AST. The simplest one is a recursive descent parser (which is easy to implement by hands).

Before start coding the parser, language grammar has to be defined. A grammar is a set of rules how tokens can be combined to make meaning. Here are the rules for Brainf*ck:

- STATEMENTS -&gt; STATEMENT STATEMENTS &#124; Epsilon
- STATEMENT1 -&gt; left&#124;right&#124;up&#124;down&#124;read&#124;write&#124;WHILE
- WHILE -&gt; while_start STATEMENTS while_end

And few notes about the grammar (just on a high level, there is a really large volume of theory behind)

- Epsilon is a special symbol, it means empty or null
- Separators on right side mean OR (one of)
- Rules from above are productions; they explain how a left side may be presented. For example, STATEMENTS is either null or a STATEMENT, followed by another STATEMENTS (welcome the recursion); and STATEMENT is one of specified terms.
- Lower case terms are terminals, they can not be on a left side
- Upper case terms are non-terminals, they are on a left side, and they end up extended to all terminals

These rules are easy to implement, by creating one function per rule. [Here is a possible implementaion.](https://github.com/andrewromanenco/brainfk-js/blob/master/js/parser.js)

The outcome of a successful parsing is an AST. It’s built while walking through grammar rules. For example, it might look like this: ```{op:left, next:{op:while, statements:{op:up, next:null} ,next{op:right, next:null}:}}``` - this represents operation left, followed by while, followed by operation right. And while has single operation ‘up’ in its body.

### Generate it

Having a valid AST is a big milestone. There are two general use case for an AST:

- walk the tree and execute all operations
- generate some kind of intermediary representation or byte code to use in other layers

[This source does code generation according to VM specs.](https://github.com/andrewromanenco/brainfk-js/blob/master/js/compiler.js) It implements a visitor pattern; and for every node in AST, the code will generate a byte code step. The result is runnable on the VM built before!

## Summary
It works!

Source code -&gt; List of tokens -&gt; Abstract syntax tree -&gt; VM instruction set -&gt; VM execution

## Next...

 - Read The Dragon Book
 - Join Compilers course at coursera

---
layout: post
title:  "Finding out most common passwords with AWS (and golang)"
date:   2016-11-16
tags: [cryptography, stuff]
---

Recently, I read a comment at stackoverflow that a computer with a ssh port open to the Internet sees attempts to login within first hour. This triggered an idea...

Simple search for "most common password" reveals a list:

 - 123456
 - 123456789
 - qwerty
 - password

The hypothesis is that any attacker should try passwords ordered by probability how often they are used. To check this hypothesis, I am going to get a server in AWS and open SSH port to public. And the SSH service is going to log all attempts with password related data. Ideally, the result should contain a list of most common passwords from "experts".

## Steps

  - Write a custom SSH server to log required data (golang)
  - Configure a server at AWS
  - Collect&analyze data

## Custom SSH server

Working Go application is available here: <a href="https://github.com/andrewromanenco/sshlistener" target="_blank">https://github.com/andrewromanenco/sshlistener</a>.

High level design for the application can be described with this statement: the app is listening a port and for every incoming connection new goroutine is stared to handle login and password data. Goroutines collect data and send events to a channel to be logged in another goroutine.

### Best practices:

  - Don’t use global variables, otherwise this is going to be a testing nightmare
  - Goroutines are regular functions run in a separate thread. While coding, treat them as regular function, e.g. channels are just parameters.
  - Channels may be used in single threaded flows. For example, in tests.

Golang already has a ssh package (golang.org/x/crypto/ssh) with both server and client implementations; and for a custom server just small pieces of code should be provided.

### Handle authentication

SSH server works with a provided connection and handles authentication via a callback. In this usecase, the callback listens for login password information and replies with "access denied". The provided data is sent as a string event to the logging channel.

The only catch with callback function is its interface. In this case, there is no good way to set a channel in the callback via func parameters.

One way of doing this would be to use a global variable. This way is not great because of testing issues; it will be quite messy to test the code properly.

Fortunately,  closure is much better approach. Here is how a closure for a callback may be defined: <a href="https://github.com/andrewromanenco/sshlistener/blob/master/sshlistener.go#L25" target="_blank">sshlistener.go#L25</a>.

{% highlight go %}
type pwdCallback func(c ssh.ConnMetadata, pass []byte) (*ssh.Permissions, error)

func pwdCallbackFactory(ch chan<- string) pwdCallback {
  return func(c ssh.ConnMetadata, pass []byte) (*ssh.Permissions, error) {
    ...
    ch <- entry
    ...
  }
}
{% endhighlight %}


With closure, the code is very testable as we can supply stub/mock: <a href="https://github.com/andrewromanenco/sshlistener/blob/master/sshlistener_test.go#L14" target="_blank">example</a>. As a result, our callback is fully tested in right context.


### Handle logging

Logging is done via a dedicated goroutine. This eliminates all data races and let the exact logging code to evolve independently. For the entire application, the interface to a log is a channel accepting strings.
<a href="" target="_blank"></a>https://github.com/andrewromanenco/sshlistener/blob/master/sshlistener.go#L36

{% highlight go %}
func writeToFile(ch <-chan string, filePath string) {
    ...
}
{% endhighlight %}


### Wire everything together

With all functions in place (and tested) the rest of the code is pretty straightforward: <a href="https://github.com/andrewromanenco/sshlistener/blob/master/sshlistener.go#L77" target="_blank">runServer(...)</a>.

  - Start listening a port
  - Run logging goroutine
  - For every incoming connection, run goroutine to handle authentication

The minor optimization is to limit number of concurrent connections. This is quite easy to implement with go's buffered channels.

## Configuring a server with AWS

Arguably, AWS is the best tool for experimenting. To run this exercise, a server has to be created and properly configured. Here is step-by-step instruction:

  1. Get an AWS account
  2. Go to EC2 and start an instance. Ubuntu 16.04 is good (you can find it in the market place). I used medium instance, but any other will work as well.
  3. During configuration process, AWS will ask you to pick a key. This is a key to be used for SSH authentication to login into the server. If there are no keys yet, pick create; the AWS will provide you a private key which you should save to your local computer. Also, don't forget to let AWS to pick a public ip for your instance.
  4. At this moment, the server is up and running; and standard SSH server is listening port 22. We need to change this. We want to move standard SSH to a different port to let our golang app to bind on 22.
  - In AWS console, click on your instance to see config data. Find "Security groups" under description tab and go there. This is the configuration for the firewall. Add custom TCP inbound connection and set port to 2022 and source to everywhere. Make sure you save all changes!
  - Now, in the AWS console use right click on your instance and click connect. This will give you exact command to run in the console to connect to your server.
  - Connect the server and edit ```/etc/ssh/sshd_config``` to change Port from 22 to 2022. Restart the server.
  - Login back using new port: append to the ssh command -p2022
  5. Install golang by running: ```sudo add-apt-repository ppa:ubuntu-lxc/lxd-stable```, ```sudo apt-get update``` and ```sudo apt-get install golang```
  6. ```export $GOPATH=/home/ubuntu/```
  7. Clone the app (or upload your own code): ```git clone https://github.com/andrewromanenco/sshlistener```
  8. ```cd  sshlistener``` and run ```go get``` followed by ```go build .```
  9. Now you have everything ready and run the app in the background: ```sudo /home/ubuntu/sshlistener/sshlistener -private=id_rsa -output=log.out -port=22 &```
     Notes:
     - id_rsa is a private key, generated by ssh-gen
     - running code under root is not the best practice, but it's ok for this experiment.

  10. Wait for connections and see entries in log.out file


## Collect and analyze data

  - It took 23 minutes before first login attempt
  - The most popular password is '123456' indeed
  - The second most popular is 'root'.
  - Third place is taken by 'password' and 'Asdf1234'
  - 99% of attempts use username 'root'

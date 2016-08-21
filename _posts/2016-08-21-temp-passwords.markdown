---
layout: post
title:  "2-factor auth and Google Authenticator app"
date:   2016-08-21
tags: [cryptography]
---

The second biggest risk in using a password is getting the password stolen. The first biggest risk in using a password is getting it stolen and not knowing about this fact. There are several ways to mitigate this type of risk, some of those methods require hardware solutions and some are 100% software.

Using a temporary password or token in addition to a regular login is the way to improve security a lot. A great example of this idea implementation is 2-factor authentication for Google products. After a user enters his/her login and password, the server requests an additional piece of authentication - a temporarily token which is valid for 30 seconds and keep changing.

Algorithms for temporary tokens generation are described in RFC 6238 and RFC 4226.The top-level overview for temporary password validation is described in these steps:

 - client and server agree on a shared secret
 - at every given point of time, both server and client may run generation algorithm and the output will match
 - running generation algorithm at different times will generate different outputs

The generation algorithm for temporary password generation is a function:
{% highlight go %}
temp_password := generate(time, secret)
{% endhighlight %}


Although the algorithm function looks trivial, it still has several points to consider.

What is time? Obviously that this can not be the current time.  A client and a server may be located in different time zones. This can be resolved by using UTC time, but this does require additional processing. The best solution for time is Unix timestamp. This value is a number and it does not depend on time zone - after all, this is just a count of seconds since Unix epoch (Thursday, 1 January 1970).

A temporary password is temporal. The question is for how long? Default time window is 30 seconds.

A shared secret is just a key of x bits. RFC recommend various length, but Google Authenticator uses 80 bit one. When a key is generated, it is important to use a cryptographic random generator and not something like Math.rand(). To make a key user-friendly, it should be encoded with BASE32 before showing to a user.

{% highlight go %}
// GenerateKey generates random crypto key of requested length in bytes.
func GenerateKey() ([]byte, error) {
  key := make([]byte, googleAuthenticatorKeySize)
  if _, err := io.ReadFull(rand.Reader, key); err != nil {
    return nil, err
  }
  return key, nil
}

// EncodeKey converts a binary key to a user friendly base32 string.
func EncodeKey(key []byte) string {
  return base32.StdEncoding.EncodeToString(key)
}
{% endhighlight %}

With a key and a timestamp value, it's time to generate a Message Authenticated Code - MAC. Strictly speaking, the algorithm is using SHA1, so the code is HMAC. The calculation is simple:
{% highlight go %}
HMAC = SHA1(concatenate(key, time/30)).
{% endhighlight %}

or, in GoLang:

{% highlight go %}
func generateHMAC(key []byte, variable int64) ([]byte, error) {
  list := bytes.Buffer{}
  err := binary.Write(&list, binary.BigEndian, variable)
  if err != nil {
    return nil, err
  }
  macProducer := hmac.New(sha1.New, key)
  macProducer.Write(list.Bytes())
  return macProducer.Sum(nil), nil
}
{% endhighlight %}

The HMAC is unique and temporal. Every 30 seconds the output is a new one. The only problem with HMAC it's long - 160 bits, not that easy to be entered by a user.
HMAC gets converted to a user-friendly code with these steps:

 - take last four bits of HMAC (this is a value in the range 0..15)
{% highlight go %}
offset := hash[19] & 0xf
{% endhighlight %}
 - treat the value from the previous step as an offset in HMAC
 - take 4 bytes from HMAC offset and convert them to a number (shift bits)
 - set sign bit to zero to make sure the number is the same everywhere
 - divide result by 1.000.000 to get a number of 6 digits max
{% highlight go %}
binCode := int32(0)
binCode += int32(hash[offset+3])
binCode += (int32(hash[offset+2]) << 8)
binCode += (int32(hash[offset+1]) << 16)
binCode += (int32(hash[offset]&0x7f) << 24)
return binCode % 1000000
{% endhighlight %}
 - if the number is shorter than 6 digits, prepend it with zeros

### Example

 - Get the code from <a href="https://github.com/andrewromanenco/g2fa" target="_blank">
github</a>
 - Add key: "PT2KHGTK7YQ3EVIK" to Google Authenticator app
 - Run code from README in console
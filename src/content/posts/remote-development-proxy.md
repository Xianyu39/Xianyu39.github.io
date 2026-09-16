---
title: "远程开发代理"
pubDate: 2025-05-22
description: "用 SSH 连接远程服务器进行开发时，可以让服务器使用本机上的代理。"
author: "Westwoods"
tags: [开发工具, SSH, 网络]
draft: false
---

用 SSH 连接远程服务器进行开发的时候经常会遇到需要访问外部网络的情况，但是在服务器上安装代理软件（比如 clash）往往不太方便，而且还会导致你要维护多个的代理。此外，如果是多用户系统的话，在服务器上启动代理也不够安全，其他用户通过 `localhost` 就能访问你的代理服务器，导致你的流量可能为大家所共享。

为了解决这个问题，我们可以考虑**让服务器使用本机上的代理**。这一共有两种方式。
# LAN
打开 Clash 的 LAN 选项，允许代理端口被远程接入，这样服务器就可以把流量发回本机的代理程序。使用 ssh 登录后，`SSH_CLIENT` 这一环境变量会记录本机 ip 地址，因此只需要将代理设置为：
```shell
export http_proxy=http://$SSH_CLIENT:7890
export https_proxy=http://$SSH_CLIENT:7890
```
即可。

但是这有一个问题：你的代理程序端口暴露在子网上，而且没有任何鉴权机制，可能会引发安全问题。
# SSH 隧道
SSH 有建立隧道的功能，即允许本机和服务器建立安全的连接，互相映射端口。我们可以**将本机的 7890 端口通过 SSH 映射到服务器上**来实现代理。

SSH 有 `-L` 和 `-R` 两个参数，这两个参数都可以进行端口转发，但是作用有不同：
1. `-L`：local，把服务器的端口转发到本地； `ssh -L port1:localhost:port2` 的意思是将服务器的 `port2` 映射到本地 `port1`，是**将本机流量转发到服务器**；
2. `-R`：remote，把本地的端口转发到服务器； `ssh -R port1:localhost:port2` 的意思是将本机的 `port2` 映射到服务器的 `port1`，是**将服务器流量转发到本机**。

> 这个端口参数看起来很奇怪，可以记忆为 `[src_ip:src_port]:[dest_ip:dest_port]`。SSH 做的始终都是将 `src` 处的流量转发到 `dest` 处。

于是我们可以修改一下登录命令：
```shell
ssh -R 7890:localhost:7890 user@server
```
这样登录的同时就完成了端口映射。

多个端口映射还可以叠用，比如：
```shell
ssh  -L 8888:localhost:9000 -R 7890:localhost:7890 user@server
```
# 调用
可以在 `~/.bashrc` 中添加命令来方便启动和关闭代理：
```shell
alias proxyon='export http_proxy=http://localhost:7890 && export https_proxy=http://localhost:7890'
alias proxyoff='unset http_proxy && unset https_proxy'
```

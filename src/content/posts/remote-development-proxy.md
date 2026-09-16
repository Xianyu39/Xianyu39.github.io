---
title: "用 SSH 反向隧道共享本机代理"
pubDate: 2026-09-17
description: "远程开发时，让服务器通过 SSH 安全使用本机代理，并说明 LAN 暴露和端口转发的区别。"
author: "Westwoods"
tags: [开发工具, SSH, 网络]
draft: false
---

远程服务器经常需要访问代码仓库、模型仓库或软件源。直接在服务器上安装代理会增加维护成本，多用户机器还可能把代理端口暴露给其他用户。一个更安全的思路是：代理运行在本机，服务器通过 SSH 隧道把流量转回本机。

## 方案一：LAN 代理

如果本机代理软件允许局域网连接，可以在服务器上把代理地址指向 SSH 客户端地址：

```bash
export http_proxy="http://${SSH_CLIENT%% *}:7890"
export https_proxy="http://${SSH_CLIENT%% *}:7890"
```

这种方式配置简单，但代理端口会暴露在局域网中。除非明确配置访问控制和鉴权，否则不建议在公共网络或多人环境中使用。

## 方案二：SSH 反向隧道

通过 `-R`，可以把本机的代理端口映射到服务器：

```bash
ssh -R 7890:localhost:7890 user@server
```

登录后，服务器访问自己的 `localhost:7890`，实际流量会通过 SSH 隧道到达本机的 7890 端口。服务器上的环境变量可以这样设置：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export no_proxy=127.0.0.1,localhost
```

如果需要同时转发其他端口，可以叠加参数，例如：

```bash
ssh -L 8888:localhost:9000 -R 7890:localhost:7890 user@server
```

## 常见问题

`-L` 是把远端服务映射到本机，`-R` 是把本机服务映射到远端。隧道断开后代理自然失效，可以使用 `ServerAliveInterval` 和 `ServerAliveCountMax` 减少长连接无声断开的情况。

不要把代理端口绑定到 `0.0.0.0`，不要把 token 或密码写入脚本，也不要在不可信服务器上使用未经审计的代理配置。完成工作后可以执行：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

# Evidence

- 截图工具探测顺序: gnome-screenshot -> scrot -> spectacle -> import
- 本机实测: gnome-screenshot/scrot/spectacle 缺失；ImageMagick `import` 可用
- Wayland 限制: `import -window root` 与 `ffmpeg x11grab` 对本机 Xwayland root 均不可用
- 采用方案: 启动真实 xterm 运行真实命令，再 `import -window <window-id>` 抓取
- 不可抓取时: 显式写 SCREENSHOT_UNAVAILABLE，绝不使用合成/伪造图片

# Linux 命令速查表

## 📚 文件操作

```bash
# 列出文件
ls -la

# 切换目录
cd <dir>

# 创建目录
mkdir -p <dir>

# 删除
rm -rf <path>

# 复制
cp -r <src> <dst>

# 移动
mv <src> <dst>

# 查看文件
cat <file>
head -n 10 <file>
tail -n 10 <file>
```

---

## 📚 进程管理

```bash
# 查看进程
ps aux | grep <name>
top
htop

# 杀死进程
kill -9 <pid>

# 后台运行
nohup python train.py &
```

---

## 📚 系统监控

```bash
# 磁盘使用
df -h
du -sh <dir>

# 内存使用
free -h

# GPU使用
nvidia-smi
watch -n 1 nvidia-smi
```

---

## 📚 压缩/解压

```bash
# tar
tar -czvf archive.tar.gz <dir>
tar -xzvf archive.tar.gz

# zip
zip -r archive.zip <dir>
unzip archive.zip
```

---

## 📚 SSH

```bash
# 连接
ssh user@host

# 复制文件
scp file user@host:/path
scp -r dir user@host:/path
```

---

*Linux速查表*

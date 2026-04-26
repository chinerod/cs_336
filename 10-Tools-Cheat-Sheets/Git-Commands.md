# Git 命令速查表

## 📚 基础命令

### 配置
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 仓库操作
```bash
# 克隆
git clone <url>

# 初始化
git init
```

### 日常操作
```bash
# 查看状态
git status

# 添加文件
git add <file>
git add .

# 提交
git commit -m "message"

# 推送
git push origin main

# 拉取
git pull origin main
```

---

## 📚 分支操作

```bash
# 创建分支
git branch <branch>
git checkout -b <branch>

# 切换分支
git checkout <branch>

# 合并分支
git merge <branch>

# 删除分支
git branch -d <branch>
```

---

## 📚 高级操作

### 撤销操作
```bash
# 撤销修改
git checkout -- <file>

# 撤销commit
git reset --soft HEAD~1

# 查看历史
git log --oneline --graph
```

### 子模块
```bash
git submodule add <url> <path>
git submodule update --init --recursive
```

---

*Git速查表*

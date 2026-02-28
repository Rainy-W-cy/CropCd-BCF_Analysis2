````markdown
# 📘 Git 常用操作指南（Windows + GitHub SSH）

适用于：  
- 已配置好 SSH  
- 使用 GitHub 仓库  
- 日常上传代码 & 更新代码  

---

# 🚀 一、首次上传本地项目到 GitHub

## 1️⃣ 进入项目目录

```bash
cd "你的项目路径"
````

例如：

```bash
cd "C:\My File\project_name"
```

---

## 2️⃣ 初始化仓库（只需一次）

```bash
git init
git branch -M main
```

---

## 3️⃣ 配置 Git 身份（只需一次）

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

---

## 4️⃣ 添加文件并提交

```bash
git add .
git commit -m "Initial commit"
```

---

## 5️⃣ 连接远端仓库（只需一次）

```bash
git remote add origin git@github.com:你的用户名/仓库名.git
```

确认是否成功：

```bash
git remote -v
```

---

## 6️⃣ 推送到 GitHub

### 情况 A：远端是空仓库

```bash
git push -u origin main
```

### 情况 B：远端已有 README

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

# 🔄 二、日常更新代码（最常用）

以后每次修改代码，只需要：

```bash
git add .
git commit -m "修改说明"
git push
```

---

# 📥 三、更新前拉取最新代码（推荐）

多人协作或换电脑时：

```bash
git pull origin main
```

---

# 📊 四、常用查看命令

查看当前状态：

```bash
git status
```

查看提交记录：

```bash
git log --oneline
```

查看远端仓库：

```bash
git remote -v
```

查看当前分支：

```bash
git branch
```

---

# ⚠️ 五、常见问题解决

## 1️⃣ push 被拒绝（non-fast-forward）

```bash
git pull origin main
git push
```

---

## 2️⃣ 想强制覆盖远端（慎用）

```bash
git push --force
```

⚠️ 会覆盖远端代码。

---

## 3️⃣ 取消合并冲突

```bash
git merge --abort
```

---

# 📁 六、推荐添加 .gitignore

创建 `.gitignore` 文件，写入：

```
__pycache__/
*.pyc
.venv/
.env
node_modules/
dist/
build/
.vscode/
.idea/
```

---

# 🧠 推荐工作流程

每次写完代码：

```
修改代码
↓
git add .
↓
git commit -m "本次修改说明"
↓
git push
```

---

# 🎯 最常用三条命令

```bash
git add .
git commit -m "update"
git push
```

---

✔ 记住一句话：

> 修改 → add → commit → push



# Restoring Claude Memory After System Reflash

Claude Code stores project memory in `~/.claude/projects/` keyed by the absolute project path.
After a reflash this directory is wiped. Follow these steps to restore it.

## Steps

**1. Clone the repo**
```bash
git clone git@github.com:srijanpal07/robust-bev-perception.git ~/Repos/robust-bev-perception
cd ~/Repos/robust-bev-perception
```

**2. Restore the memory files**
```bash
mkdir -p ~/.claude/projects/-home-beast-Repos-robust-bev-perception/memory/
cp .claude/memory/*.md ~/.claude/projects/-home-beast-Repos-robust-bev-perception/memory/
```

**3. Verify**
```bash
ls ~/.claude/projects/-home-beast-Repos-robust-bev-perception/memory/
# Should show: MEMORY.md  project_context.md  feedback_code_style.md
```

Open the repo in your IDE and start a Claude Code session — it will load the full project
context, research plan, and code style preferences automatically.

## Keeping memory up to date

The memory files in `.claude/memory/` are committed to the repo. Whenever Claude updates
its memory during a session, copy the updated files back and commit:

```bash
cp ~/.claude/projects/-home-beast-Repos-robust-bev-perception/memory/*.md .claude/memory/
git add .claude/memory/
git commit -m "Update Claude memory"
git push
```

# Push and Pull from Kria board without main Git account

We encourage using `Deploy keys` in the github repo you fork.
This way you can just add your ssh-key.

For doing this you need to go to `settings` then to `security -> Deploy keys`. There you can add your Deploy key.

1. Generate ssh key as usual.
2. Copy ssh key to `Deploy keys` in Github
3. Initialize eval on the board:
```bash
eval "$(ssh-agent -s)"
```
4. Add your ssh key:
```bash
ssh-add ~/.ssh/your_ssh_key
```
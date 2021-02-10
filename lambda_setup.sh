# Sets up a fresh Lambda Cloud GPU instance with pytorch-generative.
# IMPORTANT: You may want to comment out dotfiles.
git clone https://www.github.com/EugenHotaj/dotfiles &&\
cat dotfiles/.tmux.conf >> .tmux.conf &&\
cat dotfiles/.bashrc >> .bashrc &&\
source .bashrc &&\
python3 -m pip install --upgrade pip &&\
git clone https://www.github.com/EugenHotaj/pytorch-generative &&\
python3 -m pip install -r pytorch-generative/requirements.txt

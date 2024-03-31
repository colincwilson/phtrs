# Container for global variables in string environment

epsilon = 'ϵ'  # avoid 'ε' (confusable with IPA), alternative '𝜀'
bos = '⋊'  # beginning-of-string marker, alternative '⊢'
eos = '⋉'  # end-of-string marker, alternative '⊣'


def init(config):
    """ Set globals with dictionary or module """
    global epsilon, bos, eos
    if not isinstance(config, dict):
        config = vars(config)
    if 'epsilon' in config:
        epsilon = config['epsilon']
    if 'bos' in config:
        bos = config['bos']
    if 'eos' in config:
        eos = config['eos']

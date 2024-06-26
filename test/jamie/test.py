import dill
from reggae.qtconsole import ReggaeDebugWindow
with open("pbjam_results_9267654.pkl", 'rb') as f:
    reggae = dill.load(f)

with open("session.pkl", 'rb') as f:
    session = dill.load(f)
window = ReggaeDebugWindow(reggae, session=session)
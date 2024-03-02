from crossvalidate import *
def runone(figsf,targs,wdecay,usekbreg = False, hss = [(4,2)], lrs = [6e-2], odn = f"./nn/data/together",
           epochs = 50, experimentts = experimetts, dataf = defaultdatafoneoutput):
    """
    This function generates figures for experiment losses for given learningrates and hidden layer sizes.

    Parameters:
    figsf: The folder to safe result figures in.d
    targs: The training and data processsing arguments passed to the dataf. defaultdatafoneoutput
    defaultdataf ignore the usefuture and usepart targs.
    wdecay: The L2 regularization coefficient.
    usekbreg (bool): Whether to use kb regularizer.
    hss (list<int>): Hidden layer sizes to try.
    lrs: The learning rates to try.
    odn: The processed data output directory name.
    epochs: The number of epochs to run.
    experimentts: The exxperiment types to attemp for each combination of lr and hs.
    dataf (function): The data preprocessing function.

    Returns:
    None
    """
    figsf = "figs/"+figsf
    for hs in hss:
        for lr in lrs:
            rs = []
            for x in experimentts:
                d, lastmodel = crossvalidate(odn,args[3],hs,targs,lr,epochs,x,usekbreg,wdecay = wdecay,dataf=dataf)
                print(d)
                v = d
                rs.append((x,v[1],v[2]))
            print(rs,np.mean([x[1] for x in rs]))
            print("\n".join([f"{x[0]} & {x[1]} & {x[2]}" for x in rs]))
            print(f"------------hs-------------")
            fig = plt.figure(figsize = (6.4,4.8))
            ax = fig.add_subplot()
            vs = [x[1] for x in rs]
            ax.plot(np.arange(len(vs)), vs)
            os.makedirs(figsf, exist_ok=True)
            ax.set_yticks(np.linspace(min(vs), max(vs), 20))
            ax.set_xticks(np.arange(len(rs)))
            ax.set_xticklabels([f"{x[0]}\n{np.round(x[1], decimals=3)}\n{np.round(x[2], decimals=4)}" for x in rs])
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Test MSE")
            plt.legend()
            plt.savefig(f"{figsf}/{'-'.join([str(y) for y in hs])}-{lr}.png")
            plt.show()
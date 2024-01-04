from argparse import ArgumentParser
from matplotlib.pyplot import show
from networkAlignmentAnalysis.experiments.registry import get_experiment

def create_experiment():
    """method for getting experiment"""
    parser = ArgumentParser(description=f"ArgumentParser for loading experiment constructor")
    parser.add_argument('--experiment', type=str, required=True, help='a string that defines which experiment to run')
    args = parser.parse_known_args()[0]
    return get_experiment(args.experiment, build=True)

if __name__ == '__main__':

    # Create experiment 
    exp = create_experiment()

    if exp.args.showprms:
        # Load saved experiment
        _ = exp.load_experiment()

    elif not exp.args.justplot:
        # Report experiment details
        exp.report(init=True, args=True, meta_args=True)

        # Run main experiment loop
        results, nets = exp.main()

        # Save results if requested
        if not exp.args.nosave:
            exp.save_experiment(results)

            if exp.args.save_networks:
                print("!!!!! Need to write generalizable network saving system !!!!!")
        
    else:
        # Otherwise load saved experiment
        results = exp.load_experiment()

        # Report saved experiment details
        exp.report(args=True)

    # Plot results
    if not exp.args.showprms:
        exp.plot(results)
        
        if exp.args.showall:
            show()

    



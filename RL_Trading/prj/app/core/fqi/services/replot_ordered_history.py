from matplotlib import pyplot as plt
import json


if __name__ == '__main__':
    root_path = '/home/a2a/a2a/RL_Trading/results/delta_std_pers30_optnostd_extended_long_testnoretrain/'
    path = 'validation_test_performances_complete.json'
    with open(root_path+path, 'r') as f:
        validation_test_performances = json.load(f)

    validation, test, std = zip(*sorted(zip(validation_test_performances['optimization_value'],
                                            validation_test_performances['test_value'],
                                            validation_test_performances['std']
                                            )))
    """
    plt.figure()
    plt.title("Performance Comparison Validation/Test (no retrain)")
    plt.plot(range(len(validation)),
             validation,
             label="Validation Performances")
    plt.plot(range(len(test)),
             [5*t for t in test], label="Test Performances")
    plt.xlabel("Trials")
    plt.ylabel("P&L")
    plt.legend()
    plt.savefig(f'{root_path}/PerformanceHistory(NoRetrain)_ordered.png')
    plt.close()
    
    """
    fig, ax1 = plt.subplots(figsize=(7,7))

    # Plot validation on the left y-axis
    ax1.set_title("Performance comparison validation vs. test (no retrain)")
    ax1.plot(range(len(validation)), validation, label="Validation Performances", color='blue')
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Validation P&L")
    ax1.tick_params(axis='y')

    # Create a second y-axis for test on the right
    ax2 = ax1.twinx()
    ax2.plot(range(len(test)), test, label="Test Performances", color='orange')
    ax2.set_ylabel("Test P&L")
    ax2.tick_params(axis='y')

    # Add legend manually
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True)

    plt.savefig(f'{root_path}/PerformanceHistory(NoRetrain)_ordered.png')
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 7))

    # Plot validation on the left y-axis
    ax1.set_title("Validation performance vs. std dev")
    ax1.plot(range(len(validation)), validation, label="Validation Performances", color='blue')
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Validation P&L")
    ax1.tick_params(axis='y')

    # Create a second y-axis for test on the right
    ax2 = ax1.twinx()
    ax2.plot(range(len(std)), std, label="Standard deviation", color='orange')
    ax2.set_ylabel("Standard deviation")
    ax2.tick_params(axis='y')

    # Add legend manually
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True)

    plt.savefig(f'{root_path}/StdHistory_val_ordered.png')
    plt.close()

    test, validation, std = zip(*sorted(zip(validation_test_performances['test_value'],
                                            validation_test_performances['optimization_value'],
                                            validation_test_performances['std']
                                            )))

    fig, ax1 = plt.subplots(figsize=(7, 7))

    # Plot validation on the left y-axis
    ax1.set_title("Test performance vs. std dev")
    ax1.plot(range(len(test)), test, label="Test Performances", color='blue')
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Test P&L")
    ax1.tick_params(axis='y')

    # Create a second y-axis for test on the right
    ax2 = ax1.twinx()
    ax2.plot(range(len(std)), std, label="Standard deviation", color='orange')
    ax2.set_ylabel("Standard deviation")
    ax2.tick_params(axis='y')

    # Add legend manually
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True)

    plt.savefig(f'{root_path}/StdHistory_test_ordered.png')
    plt.close()

"""
    plt.figure()
    plt.title("Validation performance vs. std dev")
    plt.plot(range(len(validation)),
             validation,
             label="Validation Performances")
    plt.plot(range(len(std)),
             [5*s for s in std], label="Standard deviation")
    plt.xlabel("Trials")
    plt.ylabel("P&L")
    plt.legend()
    plt.close()

"""
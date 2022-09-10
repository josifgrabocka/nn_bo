import numpy as np
import matplotlib.pyplot as plt

class visualize_smbo:

    def __init__(self, smbo_method, X, y, init_idxs, task_name="test", is_rank_version=True):
        self.smbo_method = smbo_method
        self.pending_evaluation_idxs = []
        self.current_evaluations_idxs = []
        self.X = X
        self.y = y
        self.task_name = task_name
        self.init_idxs = init_idxs
        self.is_rank_version = is_rank_version



    def evaluate(self, n_trials):

        n_initial_evaluations = len(self.init_idxs)
        data_size = len(self.X)

        # create plots
        self.plot_num_rows = 1
        self.plot_num_columns = 4

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})

        self.fig, self.axs = plt.subplots(self.plot_num_rows+1, self.plot_num_columns+1)
        plt.show(block=False)

        for c in range(self.plot_num_columns):
            self.axs[self.plot_num_rows, c].axis('off')

        for r in range(self.plot_num_rows+1):
            self.axs[r, self.plot_num_columns].axis('off')

        self.pending_evaluation_idxs = list(range(data_size))
        self.current_evaluations_idxs = []

        for i in range(n_initial_evaluations):
            idx = self.init_idxs[i]
            self.pending_evaluation_idxs.remove(idx)
            self.current_evaluations_idxs.append(idx)

        max_accuracy_history = [np.max(self.y[self.current_evaluations_idxs])]
        for i in range(n_trials):

            X_pen_idx = self.smbo_method.observe_and_suggest(X_obs=self.X[self.current_evaluations_idxs],
                                                             y_obs=self.y[self.current_evaluations_idxs],
                                                             X_pen=self.X[self.pending_evaluation_idxs])
            idx = self.pending_evaluation_idxs[X_pen_idx]

            print("trial=", i+1, "x=", self.X[idx], "y=", self.y[idx],
                  "accuracy_curve=", max_accuracy_history)

            self.pending_evaluation_idxs.remove(idx)
            self.current_evaluations_idxs.append(idx)
            max_accuracy_history.append(np.max(self.y[self.current_evaluations_idxs]))

            # visualize the performance
            self.visualize_performance(trial=i)

        plt.savefig("./plots/bo_poc.png", bbox_inches="tight", dpi=300)
        plt.show()

        return max_accuracy_history

    def visualize_performance(self, trial):

        fontsize = 10

        x_batch = self.smbo_method.inference_batch(batch_feasible_configs=self.X,
                                                   X_obs=self.X[self.current_evaluations_idxs],
                                                   y_obs=self.y[self.current_evaluations_idxs])
        y_mean, y_std = self.smbo_method.infer(x_batch, X_obs=self.X[self.current_evaluations_idxs[:-1]])

        row_idx = trial//self.plot_num_columns
        col_idx = trial % self.plot_num_columns

        self.axs[row_idx, col_idx].set_xticks((0.0, 1.0))

        self.axs[row_idx, col_idx].set_title(r"Trial $"+str(trial+1)+"$", fontsize=fontsize)

        if row_idx == self.plot_num_rows - 1:
            self.axs[row_idx, col_idx].set_xlabel(r"$\lambda$")
        if col_idx == 0:
            self.axs[row_idx, col_idx].set_ylabel(r"$y(\lambda)$")
            self.axs[row_idx, col_idx].set_yticks((0.0, 1.0))
        else:
            self.axs[row_idx, col_idx].set_yticks(())


        # plot the ground truth function
        self.axs[row_idx, col_idx].plot(self.X, self.y, c="k", label="Ground Truth", linewidth=1.0)
        # plot the evaluated points so far
        self.axs[row_idx, col_idx].scatter(self.X[self.current_evaluations_idxs[:-1]], self.y[self.current_evaluations_idxs[:-1]], c="b", label="Evaluated")
        self.axs[row_idx, col_idx].scatter(self.X[self.current_evaluations_idxs[-1]], self.y[self.current_evaluations_idxs[-1]], c="r", label="Next")
        # plot the posterior mean
        self.axs[row_idx, col_idx].plot(self.X.ravel(), y_mean, c="g", label=r"$f(\lambda,\lambda^*)$", linewidth=1.0)
        self.axs[row_idx, col_idx].fill_between(self.X.ravel(), y_mean - y_std, y_mean + y_std, color='g', alpha=0.4)


        if row_idx == 0 and col_idx == self.plot_num_columns - 1:
            self.axs[row_idx, col_idx].legend(loc=1, bbox_to_anchor=(2.0, 1.0), fontsize=fontsize)

        plt.draw()
        #plt.show()
        plt.pause(0.001)

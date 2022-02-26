from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from confusion_matrix import plot_confusion_matrix_from_data


def get_biggest_area(thresholds, data, type_of_curve="pr"):
    array_1 = np.array(data[0])
    array_2 = np.array(data[1])
    if "pr" in type_of_curve:
        areas = array_1 * array_2
    elif "roc" in type_of_curve:
        areas = -1 * (array_1 - 1) * array_2
    else:
        assert 0

    return thresholds[areas.argmax()]


def find_nearest(array, value, insurence=None):
    array = np.asarray(array)
    distances_from_goal = np.abs(array - value)
    idx = None
    if insurence is not None:
        sorted_distances = np.sort(distances_from_goal)
        recall = np.asarray(insurence)
        for d in sorted_distances:
            current_idx = np.where(distances_from_goal == d)[0]
            if recall[current_idx][0] > 0.5:
                idx = current_idx[0]
                break
    else:
        idx = distances_from_goal.argmin()

    return idx


def get_cost_array(target_code):
    if target_code == "0":
        array = {"CROCHET0": -.66, "NC1": 100, "NC2": 100, "NC4": 100, "NC5": 100, "NC6": 100, "OK0": -1, "VR1": 10,
                 "VR11": 10, "VR12": 10, "VR13": 10, "VR14": 20, "VR2": 20, "VR4": 10, "VR5": 50}
    elif target_code == "1":
        array = {"CROCHET0": 10, "NC1": -40, "NC2": 40, "NC4": 40, "NC5": 40, "NC6": 40, "OK0": 20, "VR1": -20,
                 "VR11": -10, "VR12": -10, "VR13": -10, "VR14": -20, "VR2": 20, "VR4": 20, "VR5": 20}
    elif target_code == "11":
        array = {"CROCHET0": 10, "NC1": 40, "NC2": 40, "NC4": 40, "NC5": 40, "NC6": 40, "OK0": 20, "VR1": 0,
                 "VR11": -10, "VR12": 10, "VR13": 10, "VR14": 20, "VR2": 20, "VR4": 20, "VR5": 20}
    elif target_code == "12":
        array = {"CROCHET0": 10, "NC1": 40, "NC2": 40, "NC4": 40, "NC5": 40, "NC6": 40, "OK0": 20, "VR1": 0,
                 "VR11": 10, "VR12": -10, "VR13": 10, "VR14": 20, "VR2": 20, "VR4": 20, "VR5": 20}
    elif target_code == "13":
        array = {"CROCHET0": 10, "NC1": 40, "NC2": 40, "NC4": 40, "NC5": 40, "NC6": 40, "OK0": 20, "VR1": 0,
                 "VR11": 10, "VR12": 10, "VR13": -10, "VR14": 20, "VR2": 20, "VR4": 20, "VR5": 20}
    elif target_code == "14":
        array = {"CROCHET0": 10, "NC1": 40, "NC2": 40, "NC4": 40, "NC5": 40, "NC6": 40, "OK0": 20, "VR1": 0,
                 "VR11": 10, "VR12": 10, "VR13": 10, "VR14": -20, "VR2": 20, "VR4": 20, "VR5": 20}
    elif target_code == "5":
        array = {"CROCHET0": 10, "NC1": 40, "NC2": 40, "NC4": 40, "NC5": -40, "NC6": 40, "OK0": 20, "VR1": 20,
                 "VR11": 10, "VR12": 10, "VR13": 10, "VR14": 20, "VR2": 20, "VR4": 20, "VR5": -20}
    elif target_code == "OK":
        array = {"CROCHET0": 0, "NC1": 100, "NC2": 100, "NC4": 100, "NC5": 100, "NC6": 100, "OK0": -1, "VR1": 10,
                 "VR11": 10, "VR12": 10, "VR13": 10, "VR14": 20, "VR2": 20, "VR4": 10, "VR5": 50}
    return array


class ModelPlot:
    def __init__(self, preds: np.ndarray, truth: np.ndarray, labels: list, ids: np.ndarray, full_truth: pd.DataFrame,
                 bins=None):
        self.class_number = preds.shape[1]
        self.error_count_to_plot = None
        self.best_cutoffs = np.ones([self.class_number])
        self.truth = truth.astype(int)
        self.full_truth = full_truth
        self.ids = ids
        self.binarized_truth = LabelBinarizer().fit_transform(self.truth)
        if self.class_number == 2:
            self.binarized_truth = np.hstack((1 - self.binarized_truth, self.binarized_truth))
        self.labels = labels
        self.bins = bins

        self.current_preds = preds
        assert self.class_number == self.current_preds.shape[1]
        assert self.current_preds.shape[0] == self.truth.shape[0]
        assert self.class_number == len(self.labels), "Class number derived from preds shape is not equal " \
                                                      "to labels lenght: {}!={}".format(self.class_number, len(labels))

    def plot_calibration_curve(self, ax1, ax2, n_bins=10):

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        for cls in range(self.class_number):
            fraction_of_positives, mean_predicted_value = calibration_curve(self.binarized_truth[:, cls],
                                                                            self.current_preds[:, cls], n_bins=n_bins)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (self.labels[cls]))
            ax2.hist(self.current_preds[:, cls], range=(0, 1), bins=n_bins,
                     label=self.labels[cls], histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Score given by model")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        return ax1, ax2

    def plot_confusion_matrix(self, ax, use_cutoffs=False):
        if use_cutoffs:
            self.best_cutoffs = np.array(self.best_cutoffs)

            full_cutoffs_mask = self.current_preds > self.best_cutoffs[np.newaxis, :]
            filtered_preds = np.array(self.current_preds)
            filtered_preds[~full_cutoffs_mask] = 0
            mask = np.count_nonzero(full_cutoffs_mask, axis=1).astype(bool)

            top_pick = np.argmax(filtered_preds[mask], axis=1)

            ax = plot_confusion_matrix_from_data(self.truth[mask], top_pick, ax, columns=self.labels)
            ax.set_title("Confusion matrix, with cutoff")
        else:
            top_pick = np.argmax(self.current_preds, axis=1)
            ax = plot_confusion_matrix_from_data(self.truth, top_pick, ax, columns=self.labels)
            ax.set_title("Confusion matrix")

        return ax

    def plot_roc_curve(self, ax):

        ax.set_title("One versus all (OVA) ROC curves")
        use_precision_instead = ["l"]  # ["O", "0"]
        for cls in range(self.class_number):
            fpr, tpr, thr = roc_curve(self.binarized_truth[:, cls], self.current_preds[:, cls])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve of {self.labels[cls]} (area = {roc_auc:.3f})')
            self.best_cutoffs[cls] = get_biggest_area(thr, [fpr, tpr], type_of_curve="roc")

            if self.labels[cls][0] in use_precision_instead:
                crochet_mask = self.full_truth.loc[self.ids] == "CROCHET0"
                if np.count_nonzero(crochet_mask == self.truth) == 0:
                    pre, rec, thr = precision_recall_curve(self.binarized_truth[~crochet_mask, cls],
                                                           self.current_preds[~crochet_mask, cls])
                else:
                    pre, rec, thr = precision_recall_curve(self.binarized_truth[:, cls],
                                                           self.current_preds[:, cls])
                self.best_cutoffs[cls] = thr[find_nearest(pre[:-1], 0.98, insurence=rec[:-1])]

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.00, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        return ax

    def plot_pr_curve(self, ax):

        ax.set_title("One versus all (OVA) Precision-Recall curves")

        for cls in range(self.class_number):
            pre, rec, thr = precision_recall_curve(self.binarized_truth[:, cls], self.current_preds[:, cls])
            prc_auc = auc(rec, pre)
            ax.plot(rec, pre, lw=2, label=f'PR curve of {self.labels[cls]} (area = {prc_auc:.3f})')

            if self.labels[cls][0] in ["O", "0", "C"]:
                crochet_mask = self.full_truth.loc[self.ids] == "CROCHET0"
                ok_mask = self.full_truth.loc[self.ids] == "OK0"
                # check if no overlap between crochet and category
                if np.count_nonzero(crochet_mask & self.binarized_truth[:, cls]) == 0:  # its ok only
                    pre, rec, thr = precision_recall_curve(self.binarized_truth[~crochet_mask, cls],
                                                           self.current_preds[~crochet_mask, cls])
                elif np.count_nonzero(ok_mask & self.binarized_truth[:, cls]) == 0:  # its crochet only
                    pre, rec, thr = precision_recall_curve(self.binarized_truth[~ok_mask, cls],
                                                           self.current_preds[~ok_mask, cls])
                self.best_cutoffs[cls] = thr[find_nearest(pre[:-1], 0.98, insurence=rec[:-1])]

            else:
                fpr, tpr, thr = roc_curve(self.binarized_truth[:, cls], self.current_preds[:, cls])
                self.best_cutoffs[cls] = get_biggest_area(thr, [fpr, tpr], type_of_curve="roc")
                # self.best_cutoffs[cls] = self.get_biggest_area(thr, [rec[:-1], pre[:-1]], type="pr")

        ax.set_xlim([0, 1.05])
        ax.set_ylim([0.00, 1.05])
        ax.set_xlabel('Recall (%)')
        ax.set_ylabel('Precision (%)')
        ax.legend(loc="lower left")
        return ax

    def plot_error_chart(self, ax, use_cutoffs=False):
        if use_cutoffs:
            self.best_cutoffs = np.array(self.best_cutoffs)
            cutoff_mask = np.any(self.current_preds > self.best_cutoffs, axis=1)
            # added to accept preds lower than max but actually highest that has surpassed threshold
            full_cutoffs_mask = self.current_preds > self.best_cutoffs[np.newaxis, :]
            filtered_preds = np.array(self.current_preds)
            filtered_preds[~full_cutoffs_mask] = 0

        truth_score = []
        for i in range(self.current_preds.shape[0]):
            truth_score.append(self.current_preds[i, self.truth[i]])

        df = pd.DataFrame({
            "id": self.ids,
            'Truth': self.truth,
            "Full_truth": self.full_truth.loc[self.ids],
            "truth_score": truth_score,
            'Pred': np.argmax(self.current_preds, axis=1),
            'predicted_score': np.max(self.current_preds, axis=1),
            "was_right": self.truth == np.argmax(self.current_preds, axis=1),
        })
        if use_cutoffs:
            df["predicted_score"] = np.max(filtered_preds, axis=1)
            df['Pred'] = np.argmax(filtered_preds, axis=1)
            df["was_right"] = self.truth == np.argmax(filtered_preds, axis=1)
            df = df[cutoff_mask]
        types_of_errors = df.where(df["was_right"] == False).dropna(axis=0)
        # number to label
        class_indexes = [item for item in range(len(self.labels))]
        types_of_errors["Pred"].replace(class_indexes, self.labels, inplace=True)
        """
        # REMOVES CROCHET AS ERRORS IF CLASSIFIED AS OK
        try:
            types_of_errors = types_of_errors[
                (types_of_errors["Full_truth"] != "CROCHET0") | (types_of_errors["Pred"] != "OK")]
            print("CROCHET removed from errors when sorted as OK.")
        except ValueError:
            pass
        """

        error_count = types_of_errors.groupby(["Full_truth", "Pred"]).size().reset_index(name="Count") \
            .sort_values("Count", ascending=False)

        percentages = error_count["Count"].values / np.sum(error_count["Count"].values)
        error_count["%"] = percentages.round(3) * 100

        # percentages_of_pop = error_count["Count"].values / np.sum(self.full_truth[]
        if use_cutoffs is False:
            error_means = types_of_errors.groupby(["Full_truth", "Pred"]).mean().reset_index()
            error_stds = types_of_errors.groupby(["Full_truth", "Pred"]).std()
            error_count["Truth avg"] = error_means["truth_score"].values
            error_count["Truth std"] = error_stds["truth_score"].values
            error_count["Preds avg"] = error_means["predicted_score"].values
            error_count["Preds std"] = error_stds["predicted_score"].values

        error_count = error_count.round(decimals=3)

        """if use_cutoffs:
            self.error_count_to_plot = error_count #for exported errors
        ax.axis('off')"""
        if error_count.shape[0] > 35:
            error_count = error_count[0:35]
        if use_cutoffs is False:
            table = ax.table(cellText=error_count.astype(str).values, rowLabels=error_count.index, bbox=[0, 0, 2, 1],
                             colLabels=error_count.columns, fontsize=14)
            ax.set_title("Top 35 error contributions and their score")
        else:
            table = ax.table(cellText=error_count.astype(str).values, rowLabels=error_count.index, bbox=[0, 0, 0.95, 1],
                             colLabels=error_count.columns, fontsize=14)
            ax.set_title("Top 35 error contributions, with cutoff")
        table.auto_set_font_size(False)
        table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6])

        return ax

    def plot_cutoff_cost(self, ax):

        cutoffs = np.linspace(1 / (self.class_number + 1), 1, 100)
        fraction_kept = []
        costs = []
        cost_array = get_cost_array("OK")
        for cutoff in cutoffs:
            cutoff_mask = np.any(self.current_preds > [666, cutoff], axis=1)
            # I dont care about confirmed case of "the rest", so 666 it is
            frac = np.count_nonzero(self.truth[cutoff_mask]) / np.count_nonzero(self.truth)  # only count OKs
            fraction_kept.append(frac)
            if frac == 0:
                costs.append(max(costs))
                continue

            is_wrong_mask = np.argmax(self.current_preds[cutoff_mask]) != self.truth[cutoff_mask]

            cost = np.sum([cost_array[t] for t in self.full_truth[self.ids][cutoff_mask][is_wrong_mask]])
            costs.append(cost)
        ax.plot(fraction_kept, costs, label="Fraction of OKs sorted")
        ax.plot(cutoffs, costs, label="Cutoff value")

        min_frac = fraction_kept[costs.index(min(costs))]
        min_cutoff = cutoffs[costs.index(min(costs))]
        ax.annotate(f'min: {min_cutoff:.3f} cutoff\nkeeping {min_frac * 100:.1f}%',
                    xy=(min_frac, min(costs)), xycoords='data',
                    xytext=(0.8, 0.75), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                    horizontalalignment='right', verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(.1, 1, 0.1))
        ax.set_xticks(np.arange(.02, 1, 0.02), minor=True)
        ax.set_title("Cutoff optimization")
        ax.legend(loc="lower left")

        return ax

    def show_cutoffs_table(self, ax):
        df = pd.DataFrame(self.best_cutoffs, columns=["Cutoff"], index=self.labels)
        percentage_kept = []
        for i in range(self.class_number):
            max_mask = np.argmax(self.current_preds, axis=1) == i
            mask_preds = self.current_preds[:, i] > self.best_cutoffs[i]
            mask = mask_preds & max_mask
            percentage_kept.append(np.count_nonzero(mask) / np.count_nonzero(self.binarized_truth[:, i]))
        df["% of predictions kept"] = percentage_kept
        ax.axis('off')
        df = df.round(3).astype(str)
        table = ax.table(cellText=df.values, rowLabels=df.index, bbox=[0, 0, 1, 1],
                         colLabels=df.columns, fontsize=14)
        ax.set_title("Best cutoffs and the fraction kept")
        table.auto_set_font_size(False)
        # table.auto_set_column_width([0, 1, 2])

        return ax

    def print_test_results(self, dir, name):
        plt.figure(figsize=(18, 18))
        dir = Path(dir)

        ax1 = plt.subplot2grid((3, 3), (0, 0))
        ax2 = plt.subplot2grid((3, 3), (1, 0))

        ax3 = plt.subplot2grid((3, 3), (0, 1))
        ax4 = plt.subplot2grid((3, 3), (1, 1))

        ax5 = plt.subplot2grid((3, 3), (0, 2))
        ax6 = plt.subplot2grid((3, 3), (1, 2))

        ax7 = plt.subplot2grid((3, 3), (2, 0), rowspan=2)
        ax8 = plt.subplot2grid((3, 3), (2, 2))

        ax1, ax2 = self.plot_calibration_curve(ax1, ax2)
        ax3 = self.plot_confusion_matrix(ax3, False)
        ax4 = self.plot_roc_curve(ax4)
        ax5 = self.show_cutoffs_table(ax5)

        ax6 = self.plot_confusion_matrix(ax6, True)
        ax7 = self.plot_error_chart(ax7, False)
        ax8 = self.plot_error_chart(ax8, True)

        plt.suptitle(name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dir.mkdir(parents=True, exist_ok=True)  # make sure dir exists
        """
        if self.error_count_to_plot is not None:
            self.error_count_to_plot.to_csv(dir / ("ec " + str(self.best_cutoffs[0]) + name + ".csv"))
        """
        plt.savefig(str(dir / ("result sheet " + name + ".png")))
        plt.close()

    def export_results(self, file_name, model_name="model"):
        root = Path(r"D:\Prog\Projects\AceriNet\Jupyter Notebooks")
        file = Path(f"{file_name}.csv")
        column_names = [f"score_{model_name}_" + label for label in self.labels]
        output = pd.DataFrame(self.full_truth, index=self.ids, columns=["full_truth"])
        output["truth"] = self.truth
        for name, i in zip(column_names, range(len(column_names))):
            output[name] = self.current_preds[:, i]
        output.to_csv(root / file)


if __name__ == '__main__':
    print('__main__')
    # fake data
    truth = np.random.randint(0, 4, 200)
    preds = np.random.randint(0, 10000, (200, 4))
    preds = preds / np.sum(preds, axis=1).reshape(-1, 1)
    labels = ["Potato", "Apple", "Peach", "Tomato"]

    plotter = ModelPlot(preds, truth, labels)
    plotter.print_test_results("chart_plotting_test")
    plotter.plot_cutoff_effect()

    print("print_test_results is success")

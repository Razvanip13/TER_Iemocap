import numpy as np 
import scipy.stats


class KFoldLogMaster(): 
    
    def __init__(self, filename, k_folds): 
        self.__filename = filename 
        self.__folds = k_folds
        
        self.losses_train = [] 
        self.train_accuracy_list = []
        self.train_precision_list = []
        self.train_recall_list = [] 
        self.train_f1_list = [] 
        
        self.losses_val = []
        self.val_accuracy_list = [] 
        self.val_precision_list = [] 
        self.val_recall_list = [] 
        self.val_f1_list = []
        self.val_ccc = []
        
        self.losses_test = []
        self.test_accuracy_list = [] 
        self.test_precision_list = [] 
        self.test_recall_list = [] 
        self.test_f1_list = []
        self.test_ccc = []
        
    
    def add_train_scores(self, loss, accuracy, precision, recall, f1): 
        self.losses_train.append(loss)
        self.train_accuracy_list.append(accuracy)
        self.train_precision_list.append(precision)
        self.train_recall_list.append(recall)
        self.train_f1_list.append(f1)
        
    def add_val_scores(self, loss, accuracy, precision, recall, f1, ccc=None):
        self.losses_val.append(loss)
        self.val_accuracy_list.append(accuracy)
        self.val_precision_list.append(precision)
        self.val_recall_list.append(recall)
        self.val_f1_list.append(f1)
        self.val_ccc.append(ccc)
    
        
    def add_test_scores(self, loss, accuracy=None, precision=None, recall=None, f1=None, ccc=None): 
        self.losses_test.append(loss)
        self.test_accuracy_list.append(accuracy)
        self.test_precision_list.append(precision)
        self.test_recall_list.append(recall)
        self.test_f1_list.append(f1)
        self.test_ccc.append(ccc)
    
    def __mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.norm.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
    
    def write_commence_folding(self, fold_no): 
        with open(self.__filename,"a") as file: 
            file.write('------------------------------------------------------------------------\n')
            file.write(f'Training for fold {fold_no} ...\n')
    
    def write_fold_score(self, fold_no, loss, accuracy, precision, recall,f1, ccc): 
        with open(self.__filename, "a") as file: 
            file.write(f'Score for fold {fold_no}: loss of {loss}; accuracy of {accuracy*100}%; precision of {precision*100}%; recall of {recall*100}%; recall of {f1*100}%; ccc of {ccc*100};\n')
    
    def write_average_score(self, is_regrgression=False) : 
        with open(self.__filename, "a") as file: 
            file.write('------------------------------------------------------------------------\n')
            file.write('Average scores for all train folds:\n')
            file.write(f'> Accuracy: {np.mean(self.train_accuracy_list[-self.__folds:])} (+- {np.std(self.train_accuracy_list[-self.__folds:])})\n')
            file.write(f'> Precision: {np.mean(self.train_precision_list[-self.__folds:])} (+- {np.std(self.train_precision_list[-self.__folds:])})\n')
            file.write(f'> Recall: {np.mean(self.train_recall_list[-self.__folds:])} (+- {np.std(self.train_recall_list[-self.__folds:])})\n')
            file.write(f'> F1: {np.mean(self.train_f1_list[-self.__folds:])} (+- {np.std(self.train_f1_list[-self.__folds:])})\n')
            file.write(f'> Loss: {np.mean(self.losses_train)}\n')
            file.write('------------------------------------------------------------------------\n')
            file.write('Average scores for all validation folds:\n')
            file.write(f'> Accuracy: {np.mean(self.val_accuracy_list[-self.__folds:])} (+- {np.std(self.val_accuracy_list[-self.__folds:])})\n')
            file.write(f'> Precision: {np.mean(self.val_precision_list[-self.__folds:])} (+- {np.std(self.val_precision_list[-self.__folds:])})\n')
            file.write(f'> Recall: {np.mean(self.val_recall_list[-self.__folds:])} (+- {np.std(self.val_recall_list[-self.__folds:])})\n')
            file.write(f'> F1: {np.mean(self.val_f1_list[-self.__folds:])} (+- {np.std(self.val_f1_list[-self.__folds:])})\n')
            
            if is_regrgression:
                file.write(f'> CCC: {np.mean(self.val_ccc[-self.__folds:])} (+- {np.std(self.val_ccc[-self.__folds:])})\n')
            
            file.write(f'> Loss: {np.mean(self.losses_val[-self.__folds:])}\n')
            
    def write_average_score_test(self, is_regression=False):
        with open(self.__filename, "a") as file: 
            file.write('------------------------------------------------------------------------\n')
            file.write('Average scores for test:\n')
            file.write(f'> Accuracy: {np.mean(self.test_accuracy_list[-self.__folds:])} (+- {np.std(self.test_accuracy_list[-self.__folds:])})\n')
            file.write(f'> Precision: {np.mean(self.test_precision_list[-self.__folds:])} (+- {np.std(self.test_precision_list[-self.__folds:])})\n')
            file.write(f'> Recall: {np.mean(self.test_recall_list[-self.__folds:])} (+- {np.std(self.test_recall_list[-self.__folds:])})\n')
            file.write(f'> F1: {np.mean(self.test_f1_list[-self.__folds:])} (+- {np.std(self.test_f1_list[-self.__folds:])})\n')
            if is_regression:
                file.write(f'> CCC: {np.mean(self.test_ccc[-self.__folds:])} (+- {np.std(self.test_ccc[-self.__folds:])})\n')
            file.write(f'> Loss: {np.mean(self.losses_test[-self.__folds:])}\n')
            
    def write_confidence_intervals(self): 
        with open(self.__filename, "a") as file: 
            file.write('Validation\n')
            file.write('------------------------------------------------------------------------\n')
            file.write('Confidence intervals for all folds:\n')
            the_mean, left, right = self.__mean_confidence_interval(self.val_accuracy_list[-self.__folds:])
            file.write(f'> Accuracy: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.val_precision_list[-self.__folds:])
            file.write(f'> Precision: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.val_recall_list[-self.__folds:])
            file.write(f'> Recall: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.val_f1_list[-self.__folds:])
            file.write(f'> F1: mean {the_mean} interval ({left},{right}) \n')
            file.write('------------------------------------------------------------------------\n')
            
            file.write('Test\n')
            file.write('------------------------------------------------------------------------\n')
            file.write('Confidence intervals for all folds:\n')
            the_mean, left, right = self.__mean_confidence_interval(self.test_accuracy_list[-self.__folds:])
            file.write(f'> Accuracy: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.test_precision_list[-self.__folds:])
            file.write(f'> Precision: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.test_recall_list[-self.__folds:])
            file.write(f'> Recall: mean {the_mean} interval ({left},{right}) \n')
            the_mean, left, right = self.__mean_confidence_interval(self.test_f1_list[-self.__folds:])
            file.write(f'> F1: mean {the_mean} interval ({left},{right}) \n')
            file.write('------------------------------------------------------------------------\n')
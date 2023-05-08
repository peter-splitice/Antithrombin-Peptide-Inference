## Main function for implemention.

from inference import *

if __name__ == '__main__':
    pd.set_option('display.max_rows',None)
    inference('medoid_peptides.csv')
    #inference('combined_hits.csv')
    #inference('APEADQTTPEEKPAEPEPVA)
    #inference('QSPLPERQE')
    #inference('HTLGYINDNEEGPR')
    #inference('smallhits.fasta')
    #inference('longhits.fasta')
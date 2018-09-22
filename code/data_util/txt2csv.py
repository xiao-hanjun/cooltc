# -*- coding: utf-8 -*-
import numpy as np
import os.path

output_path='../dataset/Vocabulary'
upper_path = '../dataset/question_text'
file_name=['VQADatasetA_20180815/train.txt','VQADatasetA_20180815/test.txt','VQADatasetB/train.txt','VQADatasetB/test.txt']

for file_idx in range(len(file_name)):
    data_path=os.path.join(upper_path,file_name[file_idx])
    outfile=[]
    tmp=file_name[file_idx].split('/')

    outfile.append(tmp[0])
    outfile.append(tmp[1].split('.')[0])
    write_file = '_'.join(outfile)
    write_file = write_file + '.csv'
    write_file=os.path.join(output_path,write_file)
    text_file = open(write_file, "w+")
    output = '\t'.join(['vid_name','question', 'answer', 'vid_id','key'])
    output = output + '\n'
    text_file.write(output)


    for i, line in enumerate(open(data_path)):
        row_data = []
        segs = line.strip().split(',')
        video_name=segs[0]
        video_id=video_name[3:]
        question=[]
        answer=[]
        for q_num in range(5):
            # 一个问题3个正确答案，所以问题也备份3次
            q_index=1+q_num*4
            for t in range(3):
                question.append(segs[q_index])
            answer.append(segs[q_index+1])
            answer.append(segs[q_index+2])
            answer.append(segs[q_index+3])

        for record_index in range(len(question)):
            record_id=int(video_id)*100+int(record_index)
            record_id=str(record_id)
            vid_id='FRAMEQA'+record_id
            key=video_id
            output='\t'.join([video_name, str(question[record_index]),str(answer[record_index]),str(vid_id),str(key)])
            output=output+'\n'
            text_file.write(output)
    text_file.close()




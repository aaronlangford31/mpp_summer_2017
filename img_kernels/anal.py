import os

def extract_timing(file_handle):
  content = file_handle.read()
  lines = content.split('\n')
  target = lines[4]
  return float(target[23:])

def write_csv(rows):
  file = open('./anal.csv', 'w')
  file.write('ImgDim,BlockX,BlockY,Threads,Mode,Timing\n')
  for row in rows:
    file.write(row[0])
    file.write(',')
    file.write(row[1])
    file.write(',')
    file.write(row[2])
    file.write(',')
    file.write(row[3])
    file.write(',')
    file.write(row[4])
    file.write(',')
    file.write(str(row[5]))
    file.write('\n')
  file.close()

def main():
  rows = []
  for filename in os.listdir('./experiment'):
    timing_metadata = filename.split('.')
    img_dim = timing_metadata[0]
    blockx = timing_metadata[1]
    blocky = timing_metadata[2]
    threads = timing_metadata[3]
    mode = timing_metadata[4]
    f = open('./experiment/' + filename)
    timing = extract_timing(f)
    rows.append([img_dim, blockx, blocky, threads, mode, timing])
  
  write_csv(rows)

main()  

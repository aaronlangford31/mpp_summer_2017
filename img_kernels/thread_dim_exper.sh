IMG_SIZES="1024 2048 4096"
GRIDX="1 2 4 8 16"
GRIDY="1 2 4 8 16"
THREADS="32 64 128 256"
KERNELS="c m"

for SIZE in $IMG_SIZES
  do
    for X in $GRIDX
      do
        for Y in $GRIDY
          do
            for T in $THREADS
              do
                for K in $KERNELS
                  do
                    FILE="experiment/$SIZE.$X.$Y.$T.$K.txt"
                    CMD="./driver.out $SIZE $X $Y $T $K"
                    $CMD > $FILE
                  done
              done
           done
      done
  done

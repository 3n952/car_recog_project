### data directory 구조는 다음과 같아야 한다.

괄호안은 예시, 괄호 내용은 annotation(json)참고 


root: /transfg/raw_datasets  

    ㄴ 라벨링데이터  
        ㄴ 차종외관인식    
            ㄴ (교차로)   
                ㄴ ([cr06]호계사거리)  
                    ㄴ (01번)  
                        ㄴ ~~~.json, ...,  
                          
    ㄴ 원천데이터  
        ㄴ 차종외관인식    
            ㄴ (교차로)   
                ㄴ ([cr06]호계사거리)  
                    ㄴ (01번)  
                        ㄴ ~~~.jpg, ~~~.jpg,  



### 이미지 크롭해서 저장하기

 이후 datasets/custom 디렉토리 안에 크롭된 이미지들이 저장되어 있어야 함.

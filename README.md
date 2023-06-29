aws ecr get-login-password  --profile briefingaboutit | docker login  --username AWS --password-stdin 195282370032.dkr.ecr.eu-west-3.amazonaws.com

docker build . -t 195282370032.dkr.ecr.eu-west-3.amazonaws.com/briefingaboutit:0.01

docker push 195282370032.dkr.ecr.eu-west-3.amazonaws.com/briefingaboutit:0.01

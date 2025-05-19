pipeline {
    agent any
    environment {
        MLFLOW_TRACKING_URI = 'file:///home/soumik/SPE_Final_Project/mlruns'
    }
    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'github-token', url: 'https://github.com/Soumik1410/SPE_FInal_Project.git', branch: 'master'
            }
        }
        stage('Train Model') {
            steps {
                sh '/home/soumik/ml-devops-env/bin/python train.py'
            }
        }
        /*stage('Test Model') {
            steps {
                sh '/home/soumik/ml-devops-env/bin/python test.py'
            }
        }*/
	stage('Export Requirements') {
    	    steps {
        	sh '''
            	bash -c "source /home/soumik/ml-devops-env/bin/activate && pip freeze > requirements.txt"
        	'''
    	    }
	}
        stage("Build Docker Image") {
			steps {
                script {
                	sh 'docker build -t imagecaptioner_mt2024153:latest .'
		        	echo 'Docker Image successfully created.'
                }
            }
        }
		stage('Pushing Docker Image to Hub') {
	       	steps {
	            script {
	                withDockerRegistry([credentialsId: 'dockerhub-creds', url: 'https://index.docker.io/v1/']) {
					sh 'docker tag imagecaptioner_mt2024153:latest soumik1410/imagecaptioner_mt2024153:latest'
	                sh 'docker push soumik1410/imagecaptioner_mt2024153:latest'
	                }
	        	}
	        }
		}
    }
}


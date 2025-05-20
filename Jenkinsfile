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
        }/*
        stage('Test Model') {
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
        stage('Clean Requirements') {
    steps {
        script {
            sh '''
            echo "[INFO] Cleaning torch and unused packages from requirements.txt..."
            sed -i '/torch/d' requirements.txt
            sed -i '/torchaudio/d' requirements.txt
            sed -i '/torchvision/d' requirements.txt
            sed -i '/fastapi/d' requirements.txt
            sed -i '/uvicorn/d' requirements.txt
            sed -i '/+cpu/d' requirements.txt
            # Remove extra unused packages
            sed -i '/adal/d' requirements.txt
            sed -i '/alembic/d' requirements.txt
            sed -i '/altair/d' requirements.txt
            sed -i '/annotated-types/d' requirements.txt
            sed -i '/anyio/d' requirements.txt
            sed -i '/argcomplete/d' requirements.txt
            sed -i '/azure-/d' requirements.txt
            sed -i '/backports/d' requirements.txt
            sed -i '/bcrypt/d' requirements.txt
            sed -i '/blinker/d' requirements.txt
            sed -i '/boto3/d' requirements.txt
            sed -i '/botocore/d' requirements.txt
            sed -i '/cachetools/d' requirements.txt
            sed -i '/cffi/d' requirements.txt
            sed -i '/charset-normalizer/d' requirements.txt
            sed -i '/click==/d' requirements.txt
            sed -i '/cloudpickle/d' requirements.txt
            sed -i '/contextlib2/d' requirements.txt
            sed -i '/contourpy/d' requirements.txt
            sed -i '/cryptography/d' requirements.txt
            sed -i '/cycler/d' requirements.txt
            sed -i '/databricks-sdk/d' requirements.txt
            sed -i '/Deprecated/d' requirements.txt
            sed -i '/distlib/d' requirements.txt
            sed -i '/docker==/d' requirements.txt
            sed -i '/durationpy/d' requirements.txt
            sed -i '/filelock/d' requirements.txt
            sed -i '/flatbuffers/d' requirements.txt
            sed -i '/fonttools/d' requirements.txt
            sed -i '/fsspec/d' requirements.txt
            sed -i '/gast/d' requirements.txt
            sed -i '/GitPython/d' requirements.txt
            sed -i '/google-/d' requirements.txt
            sed -i '/graphene/d' requirements.txt
            sed -i '/graphql-/d' requirements.txt
            sed -i '/greenlet/d' requirements.txt
            sed -i '/gunicorn/d' requirements.txt
            sed -i '/h11/d' requirements.txt
            sed -i '/humanfriendly/d' requirements.txt
            sed -i '/idna==/d' requirements.txt
            sed -i '/importlib_metadata/d' requirements.txt
            sed -i '/isodate/d' requirements.txt
            sed -i '/jeepney/d' requirements.txt
            sed -i '/jmespath/d' requirements.txt
            sed -i '/joblib/d' requirements.txt
            sed -i '/jsonpickle/d' requirements.txt
            sed -i '/jsonschema/d' requirements.txt
            sed -i '/keras==/d' requirements.txt
            sed -i '/kiwisolver/d' requirements.txt
            sed -i '/knack/d' requirements.txt
            sed -i '/kubernetes/d' requirements.txt
            sed -i '/libclang/d' requirements.txt
            sed -i '/Mako/d' requirements.txt
            sed -i '/Markdown/d' requirements.txt
            sed -i '/markdown-it-py/d' requirements.txt
            sed -i '/matplotlib/d' requirements.txt
            sed -i '/mdurl/d' requirements.txt
            sed -i '/mlflow/d' requirements.txt
            sed -i '/mpmath/d' requirements.txt
            sed -i '/msal/d' requirements.txt
            sed -i '/msrest/d' requirements.txt
            sed -i '/namex/d' requirements.txt
            sed -i '/narwhals/d' requirements.txt
            sed -i '/ndg-httpsclient/d' requirements.txt
            sed -i '/networkx/d' requirements.txt
            sed -i '/nltk/d' requirements.txt
            sed -i '/oauthlib/d' requirements.txt
            sed -i '/openshift/d' requirements.txt
            sed -i '/opentelemetry-/d' requirements.txt
            sed -i '/optree/d' requirements.txt
            sed -i '/paramiko/d' requirements.txt
            sed -i '/pathspec/d' requirements.txt
            sed -i '/pkginfo/d' requirements.txt
            sed -i '/platformdirs/d' requirements.txt
            sed -i '/prometheus_/d' requirements.txt
            sed -i '/proto-plus/d' requirements.txt
            sed -i '/pyarrow/d' requirements.txt
            sed -i '/pyasn1/d' requirements.txt
            sed -i '/pycparser/d' requirements.txt
            sed -i '/pydantic/d' requirements.txt
            sed -i '/pydeck/d' requirements.txt
            sed -i '/Pygments/d' requirements.txt
            sed -i '/PyJWT/d' requirements.txt
            sed -i '/PyNaCl/d' requirements.txt
            sed -i '/pyOpenSSL/d' requirements.txt
            sed -i '/pyparsing/d' requirements.txt
            sed -i '/pysftp/d' requirements.txt
            sed -i '/PySocks/d' requirements.txt
            sed -i '/python-string-utils/d' requirements.txt
            sed -i '/pytz/d' requirements.txt
            sed -i '/PyYAML/d' requirements.txt
            sed -i '/referencing/d' requirements.txt
            sed -i '/regex/d' requirements.txt
            sed -i '/requests/d' requirements.txt
            sed -i '/rich/d' requirements.txt
            sed -i '/rsa/d' requirements.txt
            sed -i '/scikit-learn/d' requirements.txt
            sed -i '/scipy/d' requirements.txt
            sed -i '/seaborn/d' requirements.txt
            sed -i '/SecretStorage/d' requirements.txt
            sed -i '/smmap/d' requirements.txt
            sed -i '/sniffio/d' requirements.txt
            sed -i '/SQLAlchemy/d' requirements.txt
            sed -i '/sqlparse/d' requirements.txt
            sed -i '/starlette/d' requirements.txt
            sed -i '/sympy/d' requirements.txt
            sed -i '/tabulate/d' requirements.txt
            sed -i '/tensorboard/d' requirements.txt
            sed -i '/termcolor/d' requirements.txt
            sed -i '/threadpoolctl/d' requirements.txt
            sed -i '/toml/d' requirements.txt
            sed -i '/tqdm/d' requirements.txt
            sed -i '/tzdata/d' requirements.txt
            sed -i '/virtualenv/d' requirements.txt
            sed -i '/websocket-client/d' requirements.txt
            sed -i '/zipp/d' requirements.txt
            echo "[INFO] Cleaned requirements.txt:"
            cat requirements.txt
            '''
        		}
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
	stage('Deploy with Ansible') {
            		steps {
                		sh 'ansible-playbook -i inventory.ini playbook.yaml'
        	}
        }
    }
}


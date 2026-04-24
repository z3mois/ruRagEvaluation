add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
sub run_makeglossaries {
    my ($base_name, $path) = fileparse($_[0]);
    pushd $path;
    if ($silent) {
        system "makeglossaries -q \"$base_name\"";
    } else {
        system "makeglossaries \"$base_name\"";
    }
    popd;
}
push @generated_exts, 'glo', 'gls', 'glg';
$clean_ext .= ' %R.ist %R.xdy';
